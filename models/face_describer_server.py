import base_server


class FDServer(base_server.BaseServer):

    def __init__(self, *args, **kwargs):
        super(FDServer, self).__init__(*args, **kwargs)
