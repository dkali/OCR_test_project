using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;

namespace OCR
{
    class Program
    {
        static void Main(string[] args)
        {
            TextExtractor extracor = new TextExtractor("full_path_to_your_image");
            Image<Bgr, Byte> processed = extracor.processImage();
            CvInvoke.Imshow("Processed", processed);
            CvInvoke.WaitKey();
        }
    }
}
