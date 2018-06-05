import asyncio
import socket
from struct import pack, unpack
from collections import namedtuple
import numpy as np
from datetime import datetime


COMMAND_PORT = 51971
STREAM_PEAKS_PORT = 51972
STREAM_SPECTRA_PORT = 51973
STREAM_SENSORS_PORT = 51974

SUCCESS = 0

_LIBRARY_VERSION = '1.9.0.1'

HyperionResponse = namedtuple('HyperionResponse', 'message content')
HyperionResponse.__doc__ += "A namedtuple object that encapsulates responses returned from a Hyperion Instrument"
HyperionResponse.message.__doc__ = "A human readable string returned for most commands."
HyperionResponse.content.__doc__ = "The binary data returned from the instrument, as a bytearray"

HPeakOffsets = namedtuple('HPeakOffsets', 'boundaries delays')
HPeakOffsets.__doc__ += "A namedtuple object that contains the boundaries and delays associated with distance compensation"
HPeakOffsets.boundaries.__doc__ = "A list of region boundary edges"
HPeakOffsets.delays.__doc__ = "A list of delays in ns that are applied to region below the respective boundary edge"

NetworkSettings = namedtuple('NetworkSettings', 'address netmask gateway')


class Hyperion(object):

    def __init__(self, address: str):

        self._address = address



    def _execute_command(self, command: str, argument: str = ''):

        return HCommTCPClient.hyperion_command(self._address, command, argument)


    @property
    def serial_number(self):
        """
        The instrument serial number.
        :type: str
        """

        return str(self._execute_command('#GetSerialNumber').content)


    @property
    def library_version(self):
        """
        The version of this API library.
        :type: str
        """

        return _LIBRARY_VERSION

    @property
    def firmware_version(self):
        """
        The version of firmware on the instrument.
        :type: str
        """

        return str(self._execute_command('#GetFirmwareVersion').content)

    @property
    def fpga_version(self):
        """
        The version of FPGA code on the instrument.
        :type: str
        """
        return str(self._execute_command('#GetFPGAVersion').content)

    @property
    def instrument_name(self):
        """
        The user programmable name of the instrument (settable).
        :type: str
        """

        return str(self._execute_command('#GetInstrumentName').content)

    @instrument_name.setter
    def instrument_name(self, name: str):

        self._execute_command('#SetInstrumentName', name)

    @property
    def is_ready(self):
        """
        True if the instrument is ready for operation, false otherwise.
        :type: bool
        """
        return unpack('B', self._execute_command('#isready').content)[0] > 0


    @property
    def channel_count(self):
        """
        The number of channels on the instrument
        :type: int
        """
        return unpack('I', self._execute_command('#GetDutChannelCount').content)[0]


    @property
    def max_peak_count_per_channel(self):
        """
        The maximum number of peaks that can be returned on any channel.
        :type: int
        """
        return unpack('I', self._execute_command('#GetMaximumPeakCountPerDutChannel').content)[0]

    @property
    def available_detection_settings(self):
        """
        A tuple of all detection settings presets that are present on the instrument.
        :type: tuple of HPeakDetectionSettings
        """

        detection_settings_data = self._execute_command('#GetAvailableDetectionSettings').content

        return HPeakDetectionSettings.from_binary_data(detection_settings_data)

    @property
    def channel_detection_setting_ids(self):
        """
        A list of the detection setting ids that are currently active on each channel.
        :type: List of int
        """
        id_list = [];

        ids = self._execute_command('#GetAllChannelDetectionSettingIds').content

        for id in ids:
            id_list.append(int(id))

        return id_list

    @property
    def active_full_spectrum_channel_numbers(self):
        """
        An array of the channels for which full spectrum data is acquired. (settable)
        :type: numpy.ndarray of int
        """

        return np.frombuffer(self._execute_command('#getActiveFullSpectrumDutChannelNumbers').content, dtype=np.int32)


    @active_full_spectrum_channel_numbers.setter
    def active_full_spectrum_channel_numbers(self, channel_numbers):

        channel_string = ''

        for channel in channel_numbers:
            channel_string += '{0} '.format(channel)

        self._execute_command('#setActiveFullSpectrumDutChannelNumbers', channel_string)

    @property
    def available_laser_scan_speeds(self):
        """
        An array of the available laser scan speeds that are settable on the instrument

        :type: numpy.ndarray of int
        """

        return np.frombuffer(self._execute_command('#GetAvailableLaserScanSpeeds').content, dtype=np.int32)

    @property
    def laser_scan_speed(self):
        """
        The current laser scan speed of the instrument. (settable)

        :type: int
        """

        return unpack('I', self._execute_command('#GetLaserScanSpeed').content)[0]

    @laser_scan_speed.setter
    def laser_scan_speed(self, scan_speed: int):

        self._execute_command('#SetLaserScanSpeed', '{0}'.format(scan_speed))

    @property
    def active_network_settings(self):
        """
        The network address, netmask, and gateway that are currently active on the instrument.

        :type: NetworkSettings namedtuple
        """
        net_addresses = self._execute_command('#GetActiveNetworkSettings').content

        address = socket.inet_ntoa(net_addresses[:4])
        mask = socket.inet_ntoa(net_addresses[4:8])
        gateway = socket.inet_ntoa(net_addresses[8:12])

        return NetworkSettings(address, mask, gateway)

    @property
    def static_network_settings(self):
        """
        The network address, netmask, and gateway that are active when the instrument is in static mode. (settable)

        :type: NetworkSettings namedtuple
        """

        net_addresses = self._execute_command('#GetStaticNetworkSettings')

        address = socket.inet_ntoa(net_addresses[:4])
        mask = socket.inet_ntoa(net_addresses[4:8])
        gateway = socket.inet_ntoa(net_addresses[8:12])

        return NetworkSettings(address, mask, gateway)

    @static_network_settings.setter
    def static_network_settings(self, network_settings: NetworkSettings):

        argument = '{0} {1} {2}'.format(network_settings.address,
                                        network_settings.netmask,
                                        network_settings.gateway)

        self._execute_command('#SetStaticNetworkSettings', argument)



    @property
    def network_ip_mode(self):
        """
        The network ip configuration mode, can be dhcp or dynamic for DHCP mode, or static for static mode. (settable)
        :type: str
        """

        return str(self._execute_command('#GetNetworkIpMode').content)

    @network_ip_mode.setter
    def network_ip_mode(self, mode):

        if mode in ['Static', 'static', 'STATIC']:
            command = '#EnableStaticIpMode'
        elif mode in ['dynamic', 'Dynamic', 'DHCP', 'dhcp']:
            command = '#EnableDynamicIpMode'
        else:
            raise HyperionError('Hyperion Error:  Unknown Network IP Mode requested')

        self._execute_command(command)

    @property
    def instrument_utc_date_time(self):
        """
        The UTC time on the instrument.  If set, this will be overwritten by NTP or PTP if enabled.

        :type: datetime.datetime
        """

        date_data = self._execute_command('#GetInstrumentUtcDateTime')

        return datetime(*unpack('HHHHH', date_data))

    @instrument_utc_date_time




class HCommTCPClient(object):
    """A class that implements the hyperion communication protocol over TCP, using asynchronous IO

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

        :return: The response as a HyperionResponse namedTuple.
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
        :return: The response as a HyperionResponse namedTuple
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
        :return: The response as a HyperionResponse namedTuple
        :rtype: HyperionResponse
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


class HPeakDetectionSettings:
    """Class that encapsulates the settings that describe peak detection for a
    hyperion channel.
    """

    def __init__(self, setting_id=0, name='', description='',
                 boxcar_length=0, diff_filter_length=0,
                 lockout=0, ntv_period=0, threshold=0, mode='Peak'):
        """

        :param setting_id: The numerical index of the setting.
        :type setting_id: int
        :param name: The name of the setting.
        :type name: str
        :param description: A longer description of the use for this setting
        :type description: str
        :param boxcar_length: The length of the boxcar filter, in units of pm
        :type boxcar_length: int
        :param diff_filter_length: The length of the difference filter, in units of pm
        :type diff_filter_length: int
        :param lockout: The spectral length, in pm, of the lockout period
        :type lockout: int
        :param ntv_period: The length, in pm, of the noise threshold voltage period.
        :type ntv_period: int
        :param threshold: The normalized threshold for detecting peaks/valleys
        :type threshold: int
        :param mode: This is either 'Peak' or 'Valley'
        :type mode: str
        """

        self.setting_id = setting_id
        self.name = name
        self.description = description
        self.boxcar_length = boxcar_length
        self.diff_filter_length = diff_filter_length
        self.lockout = lockout
        self.ntv_period = ntv_period
        self.threshold = threshold
        self.mode = mode

    @classmethod
    def from_binary_data(cls, detection_settings_data):

        (setting_id, name_length) = unpack('BB', detection_settings_data[:2])
        detection_settings_data = detection_settings_data[2:]

        name = detection_settings_data[: name_length]
        detection_settings_data = detection_settings_data[name_length:]

        (description_length,) = unpack('B', detection_settings_data[0])
        detection_settings_data = detection_settings_data[1:]

        description = detection_settings_data[: description_length]

        (boxcar_length, diff_filter_length, lockout,
         ntv_period, threshold, mode) = \
            unpack('HHHHiB', detection_settings_data[description_length:(description_length + 13)])
        # Use _remainingData in case more than one preset is contained in detectionSettingsData
        remaining_data = detection_settings_data[(description_length + 13):]

        if (mode == 0):
            mode = 'Valley'
        else:
            mode = 'Peak'

        return cls(setting_id, name, description, boxcar_length, diff_filter_length,
                   lockout, ntv_period, threshold, mode), HPeakDetectionSettings.from_binary_data(remaining_data)

    def pack(self):

        if self.mode == 'Peak':
            mode_number = 1
        else:
            mode_number = 0

        pack_string = "{0} '{1}' '{2}' {3} {4} {5} {6} {7} {8}".format(
            self.setting_id, self.name, self.description, self.boxcar_length,
            self.diff_filter_length, self.lockout, self.ntv_period,
            self.threshold, mode_number)

        return pack_string

class HyperionError(Exception):
    """Exception class for encapsulating error information from Hyperion.
    """

    # changed to reflect to error codes
    def __init__(self, message):
        self.string = message

    def __str__(self):
        return repr(self.string)