
using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Kinect;

namespace LSCKinectFramework
{
    public partial class MainWindow : Window
    {
        private KinectSensor sensor;
        private WriteableBitmap imagen;
        private byte[] buffer;
        private Skeleton[] cuerpos;

        public MainWindow()
        {
            InitializeComponent();
            Loaded += VentanaCargada;
            Closing += VentanaCerrando;
        }

        private void VentanaCargada(object sender,
                                    RoutedEventArgs e)
        {
            try
            {
                if (KinectSensor.KinectSensors.Count == 0)
                {
                    EstadoKinect.Text =
                        "No se detectó ningún Kinect.";
                    return;
                }

                sensor = KinectSensor.KinectSensors[0];

                sensor.ColorStream.Enable(
                    ColorImageFormat.RgbResolution640x480Fps30);

                // Activar seguimiento del cuerpo.
                sensor.SkeletonStream.Enable();

                int ancho = sensor.ColorStream.FrameWidth;
                int alto = sensor.ColorStream.FrameHeight;

                imagen = new WriteableBitmap(
                    ancho, alto, 96, 96,
                    PixelFormats.Bgr32, null);

                buffer = new byte[ancho * alto * 4];
                cuerpos = new Skeleton[sensor.SkeletonStream.FrameSkeletonArrayLength];

                ImagenKinect.Source = imagen;

                sensor.ColorFrameReady +=
                    Sensor_ColorFrameReady;

                sensor.SkeletonFrameReady +=
                    Sensor_SkeletonFrameReady;

                sensor.Start();

                EstadoKinect.Text =
                    "Kinect conectado. Busca un cuerpo frente al sensor.";
            }
            catch (Exception ex)
            {
                EstadoKinect.Text =
                    "Error al iniciar: " + ex.Message;
            }
        }

        private void Sensor_ColorFrameReady(
            object sender,
            ColorImageFrameReadyEventArgs e)
        {
            using (ColorImageFrame frame =
                   e.OpenColorImageFrame())
            {
                if (frame == null)
                    return;

                frame.CopyPixelDataTo(buffer);

                Dispatcher.BeginInvoke(new Action(() =>
                {
                    imagen.WritePixels(
                        new Int32Rect(0, 0,
                            frame.Width, frame.Height),
                        buffer,
                        frame.Width * 4,
                        0);
                }));
            }
        }

        private void Sensor_SkeletonFrameReady(
            object sender,
            SkeletonFrameReadyEventArgs e)
        {
            using (SkeletonFrame frame = e.OpenSkeletonFrame())
            {
                if (frame == null)
                    return;

                frame.CopySkeletonDataTo(cuerpos);

                Skeleton persona = null;

                foreach (Skeleton cuerpo in cuerpos)
                {
                    if (cuerpo.TrackingState ==
                        SkeletonTrackingState.Tracked)
                    {
                        persona = cuerpo;
                        break;
                    }
                }

                if (persona == null)
                {
                    Dispatcher.BeginInvoke(new Action(() =>
                    {
                        DatosManos.Text =
                            "No se detecta un cuerpo completo. " +
                            "Ponte frente al Kinect, a unos 1.5-3 metros.";
                    }));
                    return;
                }

                Joint manoIzquierda =
                    persona.Joints[JointType.HandLeft];

                Joint manoDerecha =
                    persona.Joints[JointType.HandRight];

                string textoIzquierda =
                    DescribirMano("Izquierda", manoIzquierda);

                string textoDerecha =
                    DescribirMano("Derecha", manoDerecha);

                Dispatcher.BeginInvoke(new Action(() =>
                {
                    DatosManos.Text =
                        "CUERPO DETECTADO\n" +
                        textoIzquierda + "\n" +
                        textoDerecha;
                }));
            }
        }

        private string DescribirMano(string nombre, Joint mano)
        {
            if (mano.TrackingState == JointTrackingState.NotTracked)
                return nombre + ": no detectada";

            return string.Format(
                "{0}: X={1:F2} m | Y={2:F2} m | Z={3:F2} m ({4})",
                nombre,
                mano.Position.X,
                mano.Position.Y,
                mano.Position.Z,
                mano.TrackingState);
        }

        private void VentanaCerrando(
            object sender,
            System.ComponentModel.CancelEventArgs e)
        {
            if (sensor != null)
            {
                sensor.ColorFrameReady -=
                    Sensor_ColorFrameReady;

                sensor.SkeletonFrameReady -=
                    Sensor_SkeletonFrameReady;

                try
                {
                    sensor.Stop();
                }
                catch
                {
                    // El sensor ya podría estar detenido.
                }
            }
        }
    }
}
