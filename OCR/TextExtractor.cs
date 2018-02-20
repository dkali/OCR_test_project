using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.OCR;
using System.Drawing;

namespace OCR
{
    class TextExtractor
    {
        int mSizeX;
        int mSizeY;
        Emgu.CV.Util.VectorOfVectorOfPoint mContours;
        Mat mLoadedImage;
        Image<Bgr, Byte> mImg;

        public TextExtractor(string path)
        {
            // Load the image
            mLoadedImage = CvInvoke.Imread(path, ImreadModes.Color);
            mImg = mLoadedImage.ToImage<Bgr, Byte>();

            mSizeY = mLoadedImage.Rows;
            mSizeX = mLoadedImage.Cols;
            Console.WriteLine("Image is " + mSizeX.ToString() + "x" + mSizeY.ToString());
        }

        public int[] GetHierarchy(Mat Hierarchy, int contourIdx)
        {
            int[] ret = new int[] { };
            if (Hierarchy.Depth != Emgu.CV.CvEnum.DepthType.Cv32S)
            {
                throw new ArgumentOutOfRangeException("ContourData must have Cv32S hierarchy element type.");
            }
            if (Hierarchy.Rows != 1)
            {
                throw new ArgumentOutOfRangeException("ContourData must have one hierarchy hierarchy row.");
            }
            if (Hierarchy.NumberOfChannels != 4)
            {
                throw new ArgumentOutOfRangeException("ContourData must have four hierarchy channels.");
            }
            if (Hierarchy.Dims != 2)
            {
                throw new ArgumentOutOfRangeException("ContourData must have two dimensional hierarchy.");
            }
            long elementStride = Hierarchy.ElementSize / sizeof(Int32);
            var offset0 = (long)0 + contourIdx * elementStride;
            if (0 <= offset0 && offset0 < Hierarchy.Total.ToInt64() * elementStride)
            {
                var offset1 = (long)1 + contourIdx * elementStride;
                var offset2 = (long)2 + contourIdx * elementStride;
                var offset3 = (long)3 + contourIdx * elementStride;
                ret = new int[4];
                unsafe
                {
                    //return *((Int32*)Hierarchy.DataPointer.ToPointer() + offset);
                    ret[0] = *((Int32*)Hierarchy.DataPointer.ToPointer() + offset0);
                    ret[1] = *((Int32*)Hierarchy.DataPointer.ToPointer() + offset1);
                    ret[2] = *((Int32*)Hierarchy.DataPointer.ToPointer() + offset2);
                    ret[3] = *((Int32*)Hierarchy.DataPointer.ToPointer() + offset3);
                }
            }
            //else
            //{
            //    return new int[] { };
            //}
            return ret;
        }

        // Determine pixel intensity
        // Apparently human eyes register colors differently.
        // TVs use this formula to determine
        // pixel intensity = 0.30R + 0.59G + 0.11B
        public double ii(Point p)
        {
            if ((p.Y >= mSizeY) || (p.X >= mSizeX) ||
                (p.Y < 0) || (p.X < 0))
            {
                Console.WriteLine("pixel out of bounds (" + p.Y.ToString() + "," + p.X.ToString() + ")");
                return 0;
            }
            int pixelB = mImg.Data[p.Y, p.X, 0];
            int pixelG = mImg.Data[p.Y, p.X, 1];
            int pixelR = mImg.Data[p.Y, p.X, 2];
            return 0.30 * pixelR + 0.59 * pixelG + 0.11 * pixelB;
        }

        // Count the number of relevant siblings of a contour
        public int count_siblings(int index, Mat h_, Emgu.CV.Util.VectorOfPoint contour, bool inc_children = false)
        {
            int count = 0;
            // Include the children if necessary
            if (inc_children)
            {
                count = count_children(index, h_, contour);
            }
            else
            {
                count = 0;
            }

            // Look ahead
            int p_ = GetHierarchy(h_, index)[0];
            while (p_ > 0)
            {
                if (keep(mContours[p_]))
                {
                    count += 1;
                }
                if (inc_children)
                {
                    count += count_children(p_, h_, contour);
                }
                p_ = GetHierarchy(h_, p_)[0];
            }

            // Look behind
            int n = GetHierarchy(h_, index)[1];
            while (n > 0)
            {
                if (keep(mContours[n]))
                {
                    count += 1;
                }
                if (inc_children)
                {
                    count += count_children(n, h_, contour);
                }
                n = GetHierarchy(h_, n)[1];
            }
            return count;
        }

        // Count the number of real children
        public int count_children(int index, Mat h_, Emgu.CV.Util.VectorOfPoint contour)
        {
            int count = 0;
            // No children
            if (GetHierarchy(h_, index)[2] < 0)
            {
                return 0;
            }
            // If the first child is a contour we care about
            // then count it, otherwise don't
            if (keep(mContours[GetHierarchy(h_, index)[2]]))
            {
                count = 1;
            }
            else
            {
                count = 0;
            }

            // Also count all of the child's siblings and their children
            count += count_siblings(GetHierarchy(h_, index)[2], h_, contour, true);
            return count;
        }

        // Whether we care about this contour
        public bool keep(Emgu.CV.Util.VectorOfPoint contour)
        {
            return CvInvoke.ArcLength(contour, true) > (mSizeX + mSizeY) / 5 ? false : true;
        }

        // Get the first parent of the contour that we care about
        public int get_parent(int index, Mat h_)
        {
            int parent = GetHierarchy(h_, index)[3];
            while (parent > 0 && !keep(mContours[parent]))
            {
                parent = GetHierarchy(h_, parent)[3];
            }

            return parent;
        }

        // Quick check to test if the contour is a child
        public bool is_child(int index, Mat h_)
        {
            return (get_parent(index, h_) > 0);
        }

        public bool include_box(int index, Mat h_, Emgu.CV.Util.VectorOfPoint contour)
        {
            Console.Write(index.ToString() + ":");
            if (is_child(index, h_))
            {
                Console.Write(" Is a child");
                Console.Write(" parent " + get_parent(index, h_).ToString() + " has " + count_children(
                    get_parent(index, h_), h_, contour).ToString() + " children");
                Console.Write(" has " + count_children(index, h_, contour).ToString() + " children");
            }

            if (is_child(index, h_) && count_children(get_parent(index, h_), h_, contour) <= 2)
            {
                Console.WriteLine(" skipping: is an interior to a letter");
                return false;
            }

            if (count_children(index, h_, contour) > 2)
            {
                Console.WriteLine(" skipping, is a container of letters");
                return false;
            }
            
            Console.WriteLine(" keeping");
            return true;
        }

        public Image<Bgr, Byte> processImage()
        {
            CvInvoke.Imshow("Source", mLoadedImage);

            // Split out each channel
            Mat[] channels = mLoadedImage.Split();

            // Run canny edge detection on each channel
            Mat blueEdges = new Mat();
            Mat greenEdges = new Mat();
            Mat redEdges = new Mat();
            int lowThresh = 200;
            int highThresh = 250;
            CvInvoke.Canny(channels[0], blueEdges, lowThresh, highThresh);
            CvInvoke.Canny(channels[1], greenEdges, lowThresh, highThresh);
            CvInvoke.Canny(channels[2], redEdges, lowThresh, highThresh);

            // Join edges back into image
            CvInvoke.BitwiseOr(blueEdges, greenEdges, blueEdges);
            CvInvoke.BitwiseOr(blueEdges, redEdges, blueEdges);

            CvInvoke.Imshow("Edges", blueEdges);

            // Find mContours
            mContours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(blueEdges, mContours, hierarchy, RetrType.Ccomp, ChainApproxMethod.ChainApproxSimple, new Point(0, 0));

            int[] parents = GetHierarchy(hierarchy, 0);
            // These are the boxes that we are determining
            Emgu.CV.Util.VectorOfVectorOfPoint keepers = new Emgu.CV.Util.VectorOfVectorOfPoint();

            // For each contour, find the bounding rectangle and decide
            // if it's one we care about
            for (int index = 0; index < mContours.Size; index++)
            {
                Rectangle rect = CvInvoke.BoundingRectangle(mContours[index]);
                if (keep(mContours[index]) && include_box(index, hierarchy, mContours[index]))
                {
                    // It's a winner!
                    keepers.Push(mContours[index]);
                }
                else
                {
                    //        if DEBUG:
                    //cv2.rectangle(rejected, (x, y), (x + w, y + h), (100, 100, 100), 1)
                    //cv2.putText(rejected, str(index_), (x, y -5),
                    //            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                }
            }

            // Make a white copy of our image
            Image<Bgr, Byte> new_image = blueEdges.ToImage<Bgr, Byte>();
            new_image.SetValue(new MCvScalar(255, 255, 255));

            // For each box, find the foreground and background intensities
            for (int index = 0; index < keepers.Size; index++)
            {
                //CvInvoke.DrawContours(new_image, keepers, index, new MCvScalar(0));
                // Find the average intensity of the edge pixels to
                // determine the foreground intensity
                double foreground = 0;
                for (int pointIndex = 0; pointIndex < keepers[index].Size; pointIndex++)
                {
                    foreground += ii(keepers[index][pointIndex]);
                }

                foreground /= keepers[index].Size;
                Console.WriteLine("FG Intensity for #" + index.ToString() + " = " + foreground.ToString());

                // Find the intensity of three pixels going around the
                // outside of each corner of the bounding box to determine
                // the background intensity
                Rectangle rect = CvInvoke.BoundingRectangle(keepers[index]);
                double[] bgPixels = {
                    // bottom left corner 3 pixels
                    ii(new Point(rect.X - 1, rect.Y - 1)),
                    ii(new Point(rect.X - 1, rect.Y)),
                    ii(new Point(rect.X, rect.Y - 1)),

                    // bottom right corner 3 pixels
                    ii(new Point(rect.X + rect.Width + 1, rect.Y - 1)),
                    ii(new Point(rect.X + rect.Width, rect.Y - 1)),
                    ii(new Point(rect.X + rect.Width + 1, rect.Y)),

                    // top left corner 3 pixels
                    ii(new Point(rect.X - 1, rect.Y + rect.Height + 1)),
                    ii(new Point(rect.X - 1, rect.Y + rect.Height)),
                    ii(new Point(rect.X, rect.Y + rect.Height + 1)),

                    // top right corner 3 pixels
                    ii(new Point(rect.X + rect.Width + 1, rect.Y + rect.Height + 1)),
                    ii(new Point(rect.X + rect.Width, rect.Y + rect.Height + 1)),
                    ii(new Point(rect.X + rect.Width + 1, rect.Y + rect.Height))
                };

                // Find the median of the background
                // pixels determined above
                Array.Sort(bgPixels);
                double background = (bgPixels.Length % 2 != 0) ?
                    (double)bgPixels[bgPixels.Length / 2]
                    : ((double)bgPixels[bgPixels.Length / 2] + (double)bgPixels[bgPixels.Length / 2 - 1]) / 2;

                Console.WriteLine("BG Intensity for #" + index.ToString() + " = " + background.ToString());

                byte actFG = 0;
                byte actBG = 0;
                // Determine if the box should be inverted
                if (foreground >= background)
                {
                    actFG = 255;
                    actBG = 0;
                }
                else
                {
                    actFG = 0;
                    actBG = 255;
                }

                // Loop through every pixel in the box and color the
                // pixel accordingly
                for (int x = rect.X; x < rect.X + rect.Width; x++)
                {
                    for (int y = rect.Y; y < rect.Y + rect.Height; y++)
                    {
                        if (y >= mSizeY || x >= mSizeX)
                        {
                            Console.WriteLine("pixel out of bounds (" + y + "," + x + ")");
                            continue;
                        }
                        if (ii(new Point(x, y)) > foreground)
                        {
                            new_image.Data[y, x, 0] = actBG;
                            new_image.Data[y, x, 1] = actBG;
                            new_image.Data[y, x, 2] = actBG;
                        }
                        else
                        {
                            new_image.Data[y, x, 0] = actFG;
                            new_image.Data[y, x, 1] = actFG;
                            new_image.Data[y, x, 2] = actFG;
                        }
                    }
                }
            }
            return new_image;
        }
    }
}
