import asyncio
from struct import pack, unpack




COMMAND_PORT = 51971
STREAM_PEAKS_PORT = 51972
STREAM_SPECTRA_PORT = 51973
STREAM_SENSORS_PORT = 51974

class HCommTCPClient(object):
    """A class that implements they hyperion communication protocol over TCP, using asynchronous IO

    """


    READ_HEADER_LENGTH = 8
    WRITE_HEADER_LENGTH = 8
    RECV_BUFFER_SIZE = 4096


    def __init__(self, address : str, port = COMMAND_PORT, loop):
        """Sets up a new HCommTCPClient for connection to a Hyperion instrument.

        :param str address: IPV4 Address of the hyperion instrument
        :param int port: The port to connect.  Different ports have different functionality.  The default works for all commands.
        :param loop:  The execution loop that is used to schedule tasks.
        """
        self.address = address
        self.port = port

        self.reader = None
        self.writer = None
        self.loop = loop
        self.read_buffer = bytearray()

    async def connect(self):


        self.reader, self.writer = await asyncio.open_connection(self.address, self.port, loop = self.loop)


    async def read_data(self, data_length):
        """Asynchronously read a fixed number of bytes from the TCP connection.
        :param data_length:
        :return: data_length number of bytes in a bytearray
        :rtype: bytearray
        """
        data = self.read_buffer
        while len(data) < data_length:
            data = data + await self.reader.read(self.RECV_BUFFER_SIZE)
        data_out = data[:data_length]
        self.read_buffer = data[data_length:]

        return data_out

    async def read_response(self):
        """Asynchronously reads a hyperion formatted response from the instrument

        :return: The response in the form of a dictionary with keys "message" and "content".
        :rtype: dict
        """
        read_header = await self.read_data(self.READ_HEADER_LENGTH)
        (status, response_type, message_length, content_length) = unpack('BBHI', read_header)

        message = await self.read_data(message_length)

        content = await self.read_data(content_length)

        return dict(message=message.decode(encoding='ascii'), content=content)


    def write_command(self, command, argument = '', request_options = 0):
        """Writes a formatted command packet to the hyperion instrument

        :param str command: The command to be sent.  Must start with "#".
        :param str argument: The argument string.  More than one arguments are included in a single space-delimited string.
        :param request_options: byte flags that determine the type of data returned by the instrument.
        """
        header_out = pack('BBHI', request_options, 0, len(command), len(argument))
        self.writer.write(header_out)
        self.writer.write(command.encode(encoding='ascii'))
        self.writer.write(argument.encode(encoding='ascii'))

    async def execute_command(self, command, argument='', request_options = 0):
        """Asynchronously writes a formatted command packet to the hyperion instrument and returns the response.

        :param str command: The command to be sent.  Must start with "#".
        :param str argument: The argument string.  More than one arguments are included in a single space-delimited string.
        :param request_options: byte flags that determine the type of data returned by the instrument.
        :return: The response in the form of a dictionary with keys "message" and "content".
        :rtype: dict
        """
        if self.writer is None:
            await self.connect()

        self.write_command(command, argument, request_options)

        response = await self.read_response()
        self.last_response = response

        return response


    @classmethod
    def hyperion_command(cls, address, command, argument='', request_options=0):
        """A self contained synchronous wrapper for sending single commands to the hyperion and receiving a response.

        :param address: The instrument ipV4 address.
        :param command: The command to be sent.  Must start with "#".
        :param argument: The argument string.  If more than one, arguments are included in a single space-delimited string.
        :param request_options: Byte flags that determine the type of data returned by the instrument.
        :return: The response in the form of a dictionary with keys "message" and "content".
        :rtype: dict
        """
        loop = asyncio.get_event_loop()

        h1 = HCommTCPClient(address, COMMAND_PORT, loop)

        loop.run_until_complete(h1.execute_command(command, argument, request_options))
        h1.writer.close()

        return h1.last_response


class HCommTCPSensorStreamer(HCommTCPClient):
    """
    A Class that can stream sensor data from a hyperion instrument.
    """

    def __init__(self, address: str, loop, queue: asyncio.Queue):
        """Sets up a new streaming client for sensor data from a hyperion instrument

        :param str address:  The instrument ipV4 address
        :param loop:  The event loop that will be used for scheduling tasks.
        :param asyncio.Queue queue: A queue that can be used for transferring streamed data within the main thread.
        """
        super().__init__(address, port=STREAM_SENSORS_PORT, loop)

        self.content_length = 0
        self.data_queue = queue

    async def stream_data(self):

        if self.content_length == 0:
            read_header = await self.read_data(self.READ_HEADER_LENGTH)
            (status, response_type, message_length, self.content_length) = unpack('BBHI', read_header)

            content = await self.read_data(self.content_length)
            data = HACQSensorData(content)

            self.timestamp_init = data.header.timestampFrac * 1e-9 + data.header.timestampInt

        else:
            content = await self.read_data(self.READ_HEADER_LENGTH + self.content_length)
            data = HACQSensorData(content[self.READ_HEADER_LENGTH:])

            timestamp = data.header.timestampFrac * 1e-9 + data.header.timestampInt

        await self.data_queue.put({'timestamp': timestamp, 'data': data.data})








