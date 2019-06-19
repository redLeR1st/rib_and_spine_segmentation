#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"

#include "itkMedianImageFilter.h"

#include <vector>
#include "itksys/SystemTools.hxx"


#include "itkOtsuMultipleThresholdsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkLabelToRGBImageFilter.h"
#include "itkAdaptiveHistogramEqualizationImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"

#include <string>


//#include "QuickView.h"


const unsigned int Dimension = 3;

typedef signed short PixelType;
typedef signed short OutputPixelType;
const unsigned int OutputDimension = 2;


typedef itk::Image< PixelType, Dimension > ImageType;


typedef itk::Image< OutputPixelType, OutputDimension > Image2DType;

typedef itk::ImageSeriesReader< ImageType > ReaderType;
typedef itk::ImageSeriesWriter< ImageType, Image2DType > SeriesWriterType;
typedef itk::GDCMImageIO ImageIOType;
typedef itk::GDCMSeriesFileNames NamesGeneratorType;

typedef itk::MedianImageFilter< ImageType, ImageType > MedianFilterType;

typedef unsigned long long int uli;

int my_otsu(uli* histogram, int bins, ImageType::SizeType img_size, uli sum);

template <class myType>
void write_for_plot(myType* a, int bins, std::string filename) {
	std::string to_plot = "[";
	
	for (int i = 0; i < bins; i++) {
		to_plot += " ";
		to_plot += std::to_string(a[i]);
	}

	to_plot += " ]";

	std::ofstream out(filename);
	out << to_plot;
	out.close();
}

int main(int argc, char* argv[]){
        if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " InputDicomDirectory  OutputDicomDirectory NumOfBins" << std::endl;
                return EXIT_FAILURE;
        }

        // Read DICOM
        NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
        namesGenerator->SetInputDirectory(argv[1]);

        const ReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();
        std::size_t numberOfFileNames = filenames.size();

        ReaderType::Pointer reader = ReaderType::New();
        ImageIOType::Pointer gdcmIO = ImageIOType::New();

        reader->SetImageIO(gdcmIO);
        reader->SetFileNames(filenames);

        try {
                reader->Update();
        }
        catch (itk::ExceptionObject &excp) {
                  std::cerr << "Exception thrown while reading the image" << std::endl;
                  std::cerr << excp << std::endl;
                  return EXIT_FAILURE;
        }

        
		
		/*
		
		// Median filter
		MedianFilterType::InputSizeType radius;
		radius.Fill(1);

		MedianFilterType::Pointer medianFilter = MedianFilterType::New();
		medianFilter->SetRadius(radius);
		medianFilter->SetInput(reader->GetOutput());// igy hivatkozok a kepre
		medianFilter->Update();
		std::cout << "median" << std::endl;
		*/
		
		ImageType::IndexType index3d;
		ImageType::SizeType size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();


		int bins = std::atoi(argv[3]);
		int min_Intensity = 0;
		int max_Intensity = 2048;
		int index = 0;
		uli sum = 0;

		uli * histogram;
		histogram = new uli[bins] ();

		int by_the_way_max = -1;
		int X, Y, Z;

		short pixelIntensity = 0;
		for (int z = 0; z < size[2]; z++) {
			for (int y = 0; y < size[1]; y++) {
				for (int x = 0; x < size[0]; x++) {
					index3d[0] = x;
					index3d[1] = y;
					index3d[2] = z;
					pixelIntensity = reader->GetOutput()->GetPixel(index3d);

					if (min_Intensity <= pixelIntensity && pixelIntensity <= max_Intensity - 1) {
						index = pixelIntensity / (max_Intensity / bins);
						sum += pixelIntensity;
						histogram[index]++;
					}
				}
			}
		}

		//write_for_plot<uli>(histogram, bins, "histogram.txt");
		std::cout << "DEBUG" << std::endl;
		int threshold = my_otsu(histogram, bins, size, sum);
		std::cout << "threshold with fancy otsu(I expect 60): " << threshold << std::endl;


		using FilterType = itk::BinaryThresholdImageFilter< ImageType, ImageType >;
		FilterType::Pointer filter = FilterType::New();
		filter->SetInput(reader->GetOutput());
		filter->SetLowerThreshold(threshold);
		filter->SetUpperThreshold(9000);
		filter->SetOutsideValue(0);
		filter->SetInsideValue(2000);
		
        // Writing DICOM
        const char * outputDirectory = argv[2];
        itksys::SystemTools::MakeDirectory(outputDirectory);

        SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
        seriesWriter->SetFileName(argv[2]);

        seriesWriter->SetInput(filter->GetOutput()); // set the image to write
        seriesWriter->SetImageIO(gdcmIO);

        namesGenerator->SetOutputDirectory(outputDirectory);
        seriesWriter->SetFileNames(namesGenerator->GetOutputFileNames());
        seriesWriter->SetMetaDataDictionaryArray(reader->GetMetaDataDictionaryArray());

        try {
                seriesWriter->Update();
        }
        catch (itk::ExceptionObject & excp) {
                std::cerr << "Exception thrown while writing the series " << std::endl;
                std::cerr << excp << std::endl;

                return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;

  }


  int my_otsu(uli* histogram, int bins, ImageType::SizeType img_size, uli sum) {

	  float* probability;
	  probability = new float[bins] ();

	  float* var;
	  var = new float[bins]();

	  //Otsu's algorithm variables
	  float p1; //first value of the probability in Otsu's algorithm with t = 1
	  float q1; //first value of q qith t = 1
	  float q1prev;//previos value of q1
	  float q1next;//next value of q1 , aka q1(t+1)
	  float mu1; //average of first group, before threshold
	  float mu2;//average of second group, after threshold
	  float mu1next;//next value of the mean for group 1(mu1(t+1))
	  float mu2next;//next value of the mean for group 2(mu2(t+1))
	  float mu;//the mean gray level value of the entire image before thresholding
	  float betweenvariance = 0;//between group variance
	  float maxbetweenvariance = 0;//max between group variance
//	  float sum = 0;//sum of all histogram values to calculate the mean grey level value of the imagem values before threshholding
	  int pixelvalue = 0;//value of a pixel
	  float optimizedthresh = 0;//optimized threshhold, at the end of otsu's algorithm this will be the thresshold with the max between group vairance

	  //calculate the probability of each histogram value and store them in the probability array
	  for (int k = 0; k <= 255; k++) {
		  probability[k] = histogram[k] / float(img_size[0] * img_size[1] * img_size[2]);
	  }
	  write_for_plot<float>(probability, bins, "probability.txt");
	  
	  //Initial probability p1 which is equal to just the probability of the 0 grey value
	  p1 = probability[0];
	  //initial q which is the sum of probabilities before 1, which is equal to p1
	  q1 = p1;
	  //initial mean before the 1 has to be 0. mu(1)=0 
	  mu1 = 0;
	  //initial mean after the 0. Initially set to 0. This gets reset in the algorithm
	  mu2 = 0;
	  //mean grey level (mu) calculation
	  mu = sum / (img_size[0] * img_size[1] * img_size[2]);
	  std::cout << "\n";

	  q1prev = q1;//set previous q1, q1(t), to equal the current q1
	  for (int t = 1; t<255; t++) {
		  q1next = q1prev + probability[t + 1]; //q1next-q1(t+1)
		  mu1next = (q1prev*mu1 + (t + 1)*(probability[t + 1])) / q1next;//set mu1(t+1)
		  mu2next = (mu - q1next * mu1next) / (1 - q1next);//set mu2(t+1)
		  betweenvariance = q1prev * (1 - q1prev)*((mu1 - mu2)*(mu1 - mu2));//calculate between group variance
																			//max between group variance is initially set to 0. Change the max between group variance, and change the optimized threshold to t if the current variance is > max.
		  var[t] = betweenvariance;
		  if (betweenvariance>maxbetweenvariance) {
			  maxbetweenvariance = betweenvariance;
			  optimizedthresh = t;//set new optimized threshhold
		  }

		  q1prev = q1next;//set q1(t) to be used in the next iteration to be equal to the current q1(t+1) which is q1next
		  mu1 = mu1next;//do the same for mu1. set mu1(t) of next iteration to equal the current mu1(t+1)
		  mu2 = mu2next;//set mu2(t) of next iteration to equal the current mu2(t+1)

		  if (q1next == 0) {
			  mu1 = 0;//this eliminates divide by 0 errors because the calculate of the next mu1 would be undefend if the next value of q1 is 0, according to the otsu recursive algorithm
		  }

	  }
	  
	  write_for_plot<float>(var, bins, "var.txt");

	  std::cout << "\n";
	  std::cout << "Threshold: " << optimizedthresh;
	  std::cout << "\n";
	  std::cout << "\n";
	  std::cout << "\n";



	  return optimizedthresh;
	  
}
