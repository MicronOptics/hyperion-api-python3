import hyperion
import asyncio
import pytest
import numpy as np

from time import sleep

instrument_ip = '10.0.41.100'


@pytest.fixture(scope='module')
def test_sensors():
    sensors = [
        ['sensor_1', 'os7510', 1, 1510.0, 66.0],
        ['sensor_2', 'os7510', 1, 1530.0, 66.0],
        ['sensor_3', 'os7510', 2, 1550.0, 66.0],
        ['sensor_4', 'os7510', 2, 1570.0, 66.0]
    ]

    return sensors



def test_sensor_streamer(test_sensors):

    hyp_inst = hyperion.Hyperion(instrument_ip)



    hyp_inst.remove_sensors()

    for sensor in test_sensors:
        hyp_inst.add_sensor(*sensor)


    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=5, loop=loop)
    stream_active = True

    serial_numbers = []


    sensor_streamer = hyperion.HCommTCPSensorStreamer(instrument_ip, loop, queue)

    async def get_data():

        while True:

            sensor_data = await queue.get()
            queue.task_done()
            if sensor_data['data']:
                serial_numbers.append(sensor_data['data'].header.serial_number)
            else:
                break


    loop.create_task(get_data())

    streaming_time = 5  # seconds

    loop.call_later(streaming_time, sensor_streamer.stop_streaming)

    loop.run_until_complete(sensor_streamer.stream_data())

    hyp_inst.remove_sensors()

    assert (np.diff(np.array(serial_numbers)) == 1).all()


def test_peak_streamer():

    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=5, loop=loop)
    stream_active = True

    serial_numbers = []



    peaks_streamer = hyperion.HCommTCPPeaksStreamer(instrument_ip, loop, queue)

    async def get_data():

        while True:

            peak_data = await queue.get()
            queue.task_done()
            if peak_data['data']:
                serial_numbers.append(peak_data['data'].header.serial_number)
            else:
                break


    loop.create_task(get_data())

    streaming_time = 5 # seconds

    loop.call_later(streaming_time, peaks_streamer.stop_streaming)

    loop.run_until_complete(peaks_streamer.stream_data())

    assert (np.diff(np.array(serial_numbers)) == 1).all()


def test_spectrum_streamer():

    hyp_inst = hyperion.Hyperion(instrument_ip)


    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=5, loop=loop)
    stream_active = True

    serial_numbers = []

    spectrum_streamer = hyperion.HCommTCPSpectrumStreamer(instrument_ip, loop, queue, hyp_inst.power_cal)

    async def get_data():

        while True:

            spectrum_data = await queue.get()
            queue.task_done()
            if spectrum_data['data']:
                serial_numbers.append(spectrum_data['data'].header.serial_number)
            else:
                break

    loop.create_task(get_data())

    streaming_time = 5  # seconds

    loop.call_later(streaming_time, spectrum_streamer.stop_streaming)

    loop.run_until_complete(spectrum_streamer.stream_data())

