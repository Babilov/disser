import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

Gst.init(None)

class VideoRtspFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(VideoRtspFactory, self).__init__(**properties)
        self.launch_string = (
    'filesrc location=road2.mp4 ! decodebin ! videoconvert ! x264enc tune=zerolatency ! '
    'rtph264pay config-interval=1 name=pay0 pt=96'
)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

class GstServer:
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")

        factory = VideoRtspFactory()
        factory.set_shared(True)

        mount_points = self.server.get_mount_points()
        mount_points.add_factory("/mystream", factory)

        self.server.attach(None)

if __name__ == '__main__':
    GObject.threads_init()
    s = GstServer()
    print("RTSP сервер запущен на rtsp://localhost:8554/mystream")
    loop = GObject.MainLoop()
    loop.run()
