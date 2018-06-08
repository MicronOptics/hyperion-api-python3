import asyncio
import socket
from struct import pack, unpack, calcsize
from collections import namedtuple
import numpy as np
from datetime import datetime
from functools import partial


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

        return self._execute_command('#GetSerialNumber').content.decode()


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

        return self._execute_command('#GetFirmwareVersion').content.decode()

    @property
    def fpga_version(self):
        """
        The version of FPGA code on the instrument.
        :type: str
        """
        return self._execute_command('#GetFPGAVersion').content.decode()

    @property
    def instrument_name(self):
        """
        The user programmable name of the instrument (settable).
        :type: str
        """

        return self._execute_command('#GetInstrumentName').content.decode()

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
        A dictionary of all detection settings presets that are present on the instrument, with keys equal to the
        setting_id.
        :type: list of HPeakDetectionSettings
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

        net_addresses = self._execute_command('#GetStaticNetworkSettings').content

        address = socket.inet_ntoa(net_addresses[:4])
        mask = socket.inet_ntoa(net_addresses[4:8])
        gateway = socket.inet_ntoa(net_addresses[8:12])

        return NetworkSettings(address, mask, gateway)

    @static_network_settings.setter
    def static_network_settings(self, network_settings: NetworkSettings):

        current_settings = self.static_network_settings
        ip_mode = self.network_ip_mode


        argument = '{0} {1} {2}'.format(network_settings.address,
                                        network_settings.netmask,
                                        network_settings.gateway)

        self._execute_command('#SetStaticNetworkSettings', argument)


        if ip_mode == 'STATIC' and current_settings.address != network_settings.address:
            self._address = network_settings.address




    @property
    def network_ip_mode(self):
        """
        The network ip configuration mode, can be dhcp or dynamic for DHCP mode, or static for static mode. (settable)
        :type: str
        """

        return self._execute_command('#GetNetworkIpMode').content.decode()

    @network_ip_mode.setter
    def network_ip_mode(self, mode):

        update_ip = False
        if mode in ['Static', 'static', 'STATIC']:
            if self.network_ip_mode in ['dynamic', 'Dynamic', 'DHCP', 'dhcp']:
                update_ip = True
                new_ip = self.static_network_settings.address
            command = '#EnableStaticIpMode'
        elif mode in ['dynamic', 'Dynamic', 'DHCP', 'dhcp']:
            command = '#EnableDynamicIpMode'
        else:
            raise HyperionError('Hyperion Error:  Unknown Network IP Mode requested')

        self._execute_command(command)

        if update_ip:
            self._address = new_ip

    @property
    def instrument_utc_date_time(self):
        """
        The UTC time on the instrument.  If set, this will be overwritten by NTP or PTP if enabled.

        :type: datetime.datetime
        """

        date_data = self._execute_command('#GetInstrumentUtcDateTime').content

        return datetime(*unpack('HHHHHH', date_data))

    @instrument_utc_date_time.setter
    def instrument_utc_date_time(self, date_time: datetime):

        self._execute_command('#SetInstrumentUtcDateTime', date_time.strftime('%Y %m %d %H %M %S'))



    @property
    def ntp_enabled(self):
        """
        Boolean value indicating the enabled state of the Network Time Protocol for automatic time synchronization.
        (settable)

        :type: bool
        """

        return unpack('I', self._execute_command('#GetNtpEnabled').content)[0] > 0

    @ntp_enabled.setter
    def ntp_enabled(self, enabled: bool):

        if enabled:
            argument = '1'
        else:
            argument = '0'

        self._execute_command('#SetNtpEnabled', argument)


    @property
    def ntp_server(self):
        """
        String containing the IP address of the NTP server. (settable)
        :type: str
        """

        return self._execute_command('#GetNtpServer').content.decode()


    @ntp_server.setter
    def ntp_server(self, server_address):

        self._execute_command('#SetNtpServer', server_address)

    @property
    def ptp_enabled(self):
        """
        Boolean value indicating the enabled state of the precision time protocol.  Note that this cannot be enabled
        at the same time as NTP.  (settable)
        :type:
        """
        return unpack('I', self._execute_command('#GetPtpEnabled').content)[0] > 0

    @ptp_enabled.setter
    def ptp_enabled(self, enabled: bool):

        if enabled:
            argument = '1'
        else:
            argument = '0'

        self._execute_command('#SetPtpEnabled', argument)

    # ******************************************************************************
    # Sensors API
    # ******************************************************************************

    def add_sensor(self, name, model, channel, wavelength, calibration_factor, distance=0):
        """Add a sensor to the hyperion instrument.  Added sensors will stream data over the sensor streaming port.
        :param name: Sensor name.  This is an arbitrary string provided by user.
        :param model: Sensor model.  This must match the specific model, currently either os7510 or os7520.
        :param channel: Instrument channel on which the sensor is present.  First channel is 1.
        :param wavelength: The wavelength band of the sensor.
        :param calibration_factor: The calibration constant for the sensor.
        :param distance: Fiber length from sensor to interrogator, in meters, integer.
        :return: None
        """
        argument = '{0} {1} {2} {3} {4} {5}'.format(name, model, channel, distance, wavelength,
                                                    calibration_factor)
        self._execute_command("#AddSensor", argument)

    def get_sensor_names(self):
        """
        Get the list of user defined names for sensors currently defined on the instrument.
        :return: Array of strings containing the sensor names
        """
        response = self._execute_command('#GetSensorNames')
        if response.message == '':
            return None

        return response.message.split(' ')

    def export_sensors(self):
        """Returns all configuration data for all sensors that are currently defined on the instrument.

        :return: Array of dictionaries containing the sensor configuration
        """
        sensor_export = self._execute_command('#ExportSensors').content

        header_version, num_sensors = unpack('HH', sensor_export[:4])
        sensor_export = sensor_export[4:]
        sensor_configs = []

        for sensor_num in range(num_sensors):
            sensor_config = dict()
            sensor_config['version'], = unpack('H', sensor_export[:2])
            sensor_export = sensor_export[2:]

            sensor_config['id'] = list(bytearray(sensor_export[:16]))
            sensor_export = sensor_export[16:]

            name_length, = unpack('H', sensor_export[:2])

            sensor_export = sensor_export[2:]
            sensor_config['name'] = sensor_export[:name_length].decode()
            sensor_export = sensor_export[name_length:]

            model_length, = unpack('H', sensor_export[:2])
            sensor_export = sensor_export[2:]
            sensor_config['model'] = sensor_export[:model_length].decode()
            sensor_export = sensor_export[model_length:]

            sensor_config['channel'], = unpack('H', sensor_export[:2])
            sensor_config['channel'] += 1
            sensor_export = sensor_export[2:]

            sensor_config['distance'], = unpack('d', sensor_export[:8])

            # drop 2 bytes for reserved field
            sensor_export = sensor_export[10:]

            detail_keys = ('wavelength',
                           'calibration_factor',
                           'rc_gain',
                           'rc_thresholdHigh',
                           'rc_thresholdLow')

            sensor_details = dict(zip(detail_keys, unpack('ddddd', sensor_export[:40])))
            sensor_export = sensor_export[40:]
            sensor_config.update(sensor_details)
            sensor_configs.append(sensor_config)
        return sensor_configs

    def remove_sensors(self, sensor_names=None):
        """Removes Sensors by name

        :param sensor_names: This can be a single sensor name string or a list of sensor names strings.  If omitted,
        all sensors are removed.
        :return: None
        """

        if sensor_names is None:
            sensor_names = self.get_sensor_names()
        elif type(sensor_names) == str:
            sensor_names = [sensor_names]
        try:
            for name in sensor_names:
                self._execute_command('#removeSensor', name)

        except TypeError:
            pass

    def save_sensors(self):
        """Saves all sensors to persistent storage.

        :return: None
        """

        self._execute_command('#saveSensors')





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
            message = b''
        if status != SUCCESS:
            self.read_buffer = bytearray()
            loop = asyncio.get_event_loop()
            loop.call_exception_handler({'message':message.decode()})
            #raise (HyperionError(message))


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
        """
        A self contained synchronous wrapper for sending single commands to the hyperion and receiving a response.

        :param address: The instrument ipV4 address.
        :type address: str
        :param command: The command to be sent.  Must start with "#".
        :type command: str
        :param argument: The argument string.  If more than one, arguments are included in a single space-delimited string.
        :type argument: str
        :param request_options: Byte flags that determine the type of data returned by the instrument.
        :type request_options: int
        :return: The response as a HyperionResponse namedTuple
        :rtype: HyperionResponse
        """
        exec_loop = asyncio.get_event_loop()

        h1 = HCommTCPClient(address, COMMAND_PORT, exec_loop)

        error_report = {'status':False}

        def exception_handler(loop, context):

            error_report['status'] = True
            error_report['message'] = context['message']

        exec_loop.set_exception_handler(exception_handler)
        exec_loop.run_until_complete(h1.execute_command(command, argument, request_options))

        h1.writer.close()

        if error_report['status']:
            raise HyperionError(error_report['message'])

        return h1.last_response

class HCommTCPStreamer(HCommTCPClient):
    """
    Abstract base class for the different streaming data sources on the Hyperion (peaks, spectra, sensors)
    """

    def __init__(self, address:str, port: int, loop, queue : asyncio.Queue, data_parser = None, fast_streaming = False):
        """
        Set up a new streaming client.
        :param address: The instrument ipV4 address
        :type address: str
        :param port: TCP port that is used for streaming.
        :type port: int
        :param loop: The event loop that will be used for scheduling tasks.
        :type loop: asyncio.loop
        :param queue: asyncio.Queue queue: A queue that can be used for transferring streamed data within the thead.
        :type queue: asyncio.Queue
        :param data_parser:  Callable that takes in the binary content from the stream and returns a data dict that can
        be consumed.
        """

        super().__init__(address, port=port, loop=loop)

        self.data_queue = queue
        self.stream_active = False
        self._data_parser = data_parser or (lambda data_in: {'data': data_in})
        self._last_content_length = None
        self._fast_streaming = fast_streaming


    async def get_data(self, content_length = None):
        """
        Asynchronously retrieve streaming data, and output the parsed data dictionary.

        :param content_length: If not none, then this will avoid the added call to retrieve the read header, and will
        retrieve both the header and the content in a single call.
        :type content_length: int
        :return: parse data
        :rtype: dict
        """

        if content_length:

            content = await self.read_data(self.READ_HEADER_LENGTH + content_length)
            content = content[self.READ_HEADER_LENGTH:]

        else:

            read_header = await self.read_data(self.READ_HEADER_LENGTH)
            (status, response_type, message_length, content_length) = unpack('BBHI', read_header)
            content = await self.read_data(content_length)

            if self._fast_streaming:
                self._last_content_length = content_length


        return self._data_parser(content)

    async def stream_data(self):
        """
        This is a producer loop that initiates the connection and Streams sensor data to the data queue to be consumed
        elsewhere.
        :return: None
        """

        await self.connect()
        self.stream_active = True

        while self.stream_active:

            data_out = await self.get_data(self._last_content_length)

            await self.data_queue.put(data_out)

    def stop_streaming(self):
        """
        Stops the endless loop within the stream_data method.
        :return: None
        """

        self.stream_active = False



class HCommTCPSensorStreamer(HCommTCPStreamer):
    """
    A Class that can stream sensor data from a hyperion instrument.
    """

    def __init__(self, address: str, loop, queue: asyncio.Queue):
        """Sets up a new streaming client for sensor data from a hyperion instrument

        :param str address:  The instrument ipV4 address
        :param loop:  The event loop that will be used for scheduling tasks.
        :param asyncio.Queue queue: A queue that can be used for transferring streamed data within the main thread.
        """
        super().__init__(address,
                         port=STREAM_SENSORS_PORT,
                         loop=loop,
                         queue = queue,
                         data_parser = HACQSensorData.data_parser,
                         fast_streaming=True)


class HCommTCPPeaksStreamer(HCommTCPStreamer):
    """
    A Class that can stream peaks data from a hyperion instrument.
    """

    def __init__(self, address: str, loop, queue: asyncio.Queue):
        """Sets up a new streaming client for sensor data from a hyperion instrument

        :param str address:  The instrument ipV4 address
        :param loop:  The event loop that will be used for scheduling tasks.
        :param asyncio.Queue queue: A queue that can be used for transferring streamed data within the main thread.
        """
        super().__init__(address,
                         port=STREAM_PEAKS_PORT,
                         loop=loop,
                         queue = queue,
                         data_parser = HACQPeaksData.data_parser,
                         fast_streaming=True)


class HCommTCPSpectrumStreamer(HCommTCPStreamer):
    """
    A Class that can stream spectrum data from a hyperion instrument.
    """

    def __init__(self, address: str, loop, queue: asyncio.Queue, powercal=None):
        """Sets up a new streaming client for sensor data from a hyperion instrument

        :param str address:  The instrument ipV4 address
        :param loop:  The event loop that will be used for scheduling tasks.
        :param asyncio.Queue queue: A queue that can be used for transferring streamed data within the main thread.
        """

        #This one line saves many
        data_parser = partial(HACQSpectrumData.data_parser, powercal=powercal)

        super().__init__(address,
                         port=STREAM_SPECTRA_PORT,
                         loop=loop,
                         queue = queue,
                         data_parser = data_parser,
                         fast_streaming=True)


class HACQSensorData(object):
    """Class that encapsulates sensor data streamed from hyperion

    """
    sensor_header = namedtuple('sensor_header',
                               ['header_length',
                                'status',
                                'buffer_percentage',
                                'reserved',
                                'serial_number',
                                'timestamp_int',
                                'timestamp_frac'])

    def __init__(self, streaming_data):
        header_length = 24
        self.header = HACQSensorData.sensor_header(*unpack('HBBIQII', streaming_data[:header_length]))

        self.data = np.frombuffer(streaming_data[header_length:], dtype=np.float)

    @classmethod
    def data_parser(cls, streaming_data):
        """
        Parser that takes in data from the stream and outputs formatted data in a dictionary
        :param streaming_data: bytearray of content from Hyperion instrument
        :type streaming_data: bytearray
        :return: Data dictionary with 'timestamp' and 'data' keys.
        :rtype: dict
        """
        sensor_data = cls(streaming_data)

        timestamp = sensor_data.header.timestampFrac * 1e-9 + sensor_data.header.timestampInt

        return {'timestamp': timestamp, 'data':sensor_data.data}

class HACQPeaksData(object):

    peaks_header = namedtuple('peaks_header',
                              ['length',
                               'version',
                               'reserved',
                               'serial_number',
                               'timestamp_int',
                               'timestamp_frac'])

    def __init__(self, raw_data):

        header_length = 24

        self._header = HACQPeaksData.peaks_header(*unpack('HHIQII', raw_data[:header_length]))

        self._peak_counts = np.frombuffer(raw_data[header_length:self._header.length], dtype=np.int16)

        self.channel_boundaries = np.cumsum(self._peak_counts)

        self.data = np.frombuffer(raw_data[self._header.length:], dtype=np.float)

        channel_start = 0

        self.channel_slices = []

        for channel_end in self.channel_boundaries:

            self.channel_slices.append(self.data[channel_start:channel_end])
            channel_start = channel_end



    @classmethod
    def data_parser(cls, raw_data):

        peaks_data = cls(raw_data)

        timestamp = peaks_data._header.timestampFrac * 1e-9 + peaks_data._header.timestampInt

        return {'timestamp': timestamp, 'data':peaks_data.data, 'channel_data':peaks_data.channel_slices}



class HACQSpectrumData(object):

    spectrum_header = namedtuple('spectrum_header',
                                 ['length',
                                  'version',
                                  'reserved',
                                  'serial_number',
                                  'timestamp_int',
                                  'timestamp_frac',
                                  'start_wavelength',
                                  'wavelength_increment',
                                  'num_points',
                                  'num_channels',
                                  'active_channel_bits'])

    def __init__(self, raw_data, powercal = None):
        header_length = 48

        self._header = HACQSpectrumData.spectrum_header(*unpack('HHIQIIddIHH', raw_data[:header_length]))

        self.data = np.frombuffer(raw_data[header_length:], dtype=np.uint16).reshape((self._header.num_channels,
                                                                                      self._header.num_points))
        self.channel_map = np.zeros(self._header.num_channels)

        map_index = 0
        for channel_index in range[16]:
            if (self._header.active_channel_bits >> channel_index) & 1:
                self.channel_map[map_index] = channel_index
                map_index += 1

        if powercal:

            self.data = self._raw_spectrum_to_db(powercal)

        self.spectra = dict()

        list(map(lambda x,y: self.spectra.update({x + 1:y}), self.channel_map, self.data))

        self.spectra_header = {
            'start_wavelength' : self._header.start_wavelength,
            'wavelength_increment': self._header.wavelength_increment,
            'num_points': self._header.num_points
        }

    def _raw_spectrum_to_db(self, powercal):

        offsets = powercal.offsets[self.channel_map]
        scales = powercal.inverse_scales[self.channel_map]

        data_db = (self.data.T * scales + offsets).T

        return data_db

    @classmethod
    def data_parser(cls, raw_data, powercal=None):

        spectra_data = cls(raw_data, powercal)

        timestamp = spectra_data._header.timestampFrac * 1e-9 + spectra_data._header.timestampInt

        return {'timestamp': timestamp, 'data':spectra_data.spectra, 'header':spectra_data.spectra_header}


class HPeakDetectionSettings(object):
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
        """

        :param detection_settings_data: Byte array of detection settings in binary format as returned from instrument
        :type detection_settings_data: bytearray
        :return: All of the detection settings in detection_settings_data, parsed as HPeakDetectionSettings objects.
        Returned as a dict with the keys being the setting_id
        :rtype: dict of HPeakDetectionSettings
        """

        detection_settings = {}

        while len(detection_settings_data):

            (setting_id, name_length) = unpack('BB', detection_settings_data[:2])
            detection_settings_data = detection_settings_data[2:]

            name = detection_settings_data[: name_length].decode()
            detection_settings_data = detection_settings_data[name_length:]

            description_length = detection_settings_data[0]
            detection_settings_data = detection_settings_data[1:]

            description = detection_settings_data[: description_length].decode()

            (boxcar_length, diff_filter_length, lockout,
             ntv_period, threshold, mode) = \
                unpack('HHHHiB', detection_settings_data[description_length:(description_length + 13)])
            # In case more than one preset is contained in detectionSettingsData
            detection_settings_data = detection_settings_data[(description_length + 13):]

            if (mode == 0):
                mode = 'Valley'
            else:
                mode = 'Peak'

            detection_settings[setting_id] = (cls(setting_id, name, description, boxcar_length, diff_filter_length,
                                                  lockout, ntv_period, threshold, mode))



        return detection_settings

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