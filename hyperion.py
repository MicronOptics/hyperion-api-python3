import asyncio
from struct import pack, unpack
from collections import namedtuple
import numpy as np


COMMAND_PORT = 51971
STREAM_PEAKS_PORT = 51972
STREAM_SPECTRA_PORT = 51973
STREAM_SENSORS_PORT = 51974

SUCCESS = 0

HyperionResponse = namedtuple('HyperionResponse', 'message content')
HyperionResponse.__doc__ += "A namedtuple object that encapsulates responses returned from a Hyperion Instrument"
HyperionResponse.message.__doc__ = "A human readable string returned for most commands."
HyperionResponse.content.__doc__ = "The binary data returned from the instrument, as a bytearray"

class HCommTCPClient(object):
    """A class that implements they hyperion communication protocol over TCP, using asynchronous IO

    """


    READ_HEADER_LENGTH = 8
    WRITE_HEADER_LENGTH = 8
    RECV_BUFFER_SIZE = 4096


    def __init__(self, address : str, port, loop):
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
        """
        Open an asyncio connection to the instrument.
        :return:
        """
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

        :return: The response as a HyperionResponse Object.
        :rtype: HyperionResponse
        """
        read_header = await self.read_data(self.READ_HEADER_LENGTH)
        (status, response_type, message_length, content_length) = unpack('BBHI', read_header)

        if message_length > 0:
            message = await self.read_data(message_length)
        else:
            message = ''
        if status != SUCCESS:
            self.read_buffer = bytearray()
            raise (HyperionError(message))


        content = await self.read_data(content_length)

        return HyperionResponse(message=message.decode(encoding='ascii'), content=content)


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
        :return: The response as a HyperionResponse Object
        :rtype: HyperionResponse
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
        :return: The response as a HyperionResponse Object
        :rtype: HyerionResponse
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
        super().__init__(address, port=STREAM_SENSORS_PORT, loop=loop)

        self.content_length = 0
        self.data_queue = queue
        self.stream_active = False

    async def stream_data(self):
        """
        Streams sensor data to the data queue.  This streamlines the data retrieval, and assumes that the number of
        sensors is going to be constant.
        :return: None
        """

        await self.connect()
        self.stream_active = True

        while self.stream_active:

            try:
                if self.content_length == 0:
                    read_header = await self.read_data(self.READ_HEADER_LENGTH)
                    (status, response_type, message_length, self.content_length) = unpack('BBHI', read_header)

                    content = await self.read_data(self.content_length)
                    data = HACQSensorData(content)

                else:
                    content = await self.read_data(self.READ_HEADER_LENGTH + self.content_length)
                    data = HACQSensorData(content[self.READ_HEADER_LENGTH:])
            except:
                self.stream_active = False
                return


            timestamp = data.header.timestampFrac * 1e-9 + data.header.timestampInt

            await self.data_queue.put({'timestamp': timestamp, 'data': data.data})

    def stop_streaming(self):

        self.stream_active = False


class HACQSensorData:
    """Class that encapsulates sensor data streamed from hyperion

    """
    sensor_header = namedtuple('sensor_header',
                               'headerLength status bufferPercentage reserved serialNumber timestampInt timestampFrac')

    def __init__(self, streamingData):
        self.header = HACQSensorData.sensor_header._make(unpack('HBBIQII', streamingData[:24]))

        self.data = np.frombuffer(streamingData[self.header.headerLength:], dtype=np.float)


class HyperionError(Exception):
    """Exception class for encapsulating error information from Hyperion.
    """

    # changed to reflect to error codes
    def __init__(self, message):
        self.string = message

    def __str__(self):
        return repr(self.string)