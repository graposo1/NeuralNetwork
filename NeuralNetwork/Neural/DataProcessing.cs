using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Neural
{
    class DataProcessing
    {
        static public List<double> ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            //using (var stream = new MemoryStream())
            //{
            //    ((Image)destImage).Save(stream, System.Drawing.Imaging.ImageFormat.Png);
            //    return stream.ToArray();
            //}

            List<double> res = new List<double>();

            for (int i = 0; i < destImage.Width; i++)
            {
                for (int j = 0; j < destImage.Height; j++)
                {
                    
                    Color pixel = destImage.GetPixel(i, j);
                    //string binary = Convert.ToString(pixel.ToArgb(), 2);
                    //var bitArr = binary.Select(c => double.Parse(c.ToString())).ToArray();
                    //for(var b = 0; b<bitArr.Length; b++)
                    //    res.Add(bitArr[b]);

                    res.Add(pixel.ToArgb() * 0.0000001);
                }
            }

            return res;
        }
    }
}
