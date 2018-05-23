import hyperion

instrument_ip = '10.0.41.90'

def test_hyperion_tcp_comm():

    response = hyperion.HCommTCPClient.hyperion_command(instrument_ip, "#GetSerialNumber")

    assert response.content[:3].decode() == 'HIA'
    assert len(response.content) == 6
