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
#include "itkVotingBinaryIterativeHoleFillingImageFilter.h"

#include "itkBinaryMorphologicalOpeningImageFilter.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

#include <string>
#include <chrono>


#include "itkImageRegistrationMethod.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

#include "itkCenteredTransformInitializer.h"
#include "itkVersorRigid3DTransform.h"
#include "itkAffineTransform.h"
#include "itkBSplineTransform.h"
#include "itkRegularStepGradientDescentOptimizer.h"

#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"
#include "itkSqrtImageFilter.h"
#include "itkTransformFileWriter.h"

#include "itkCommand.h"



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

using BinThFilterType = itk::BinaryThresholdImageFilter< ImageType, ImageType >;
using VotBinFillHoleFilterType = itk::VotingBinaryIterativeHoleFillingImageFilter< ImageType >;

int my_otsu(uli* histogram, int bins, ImageType::SizeType img_size, uli sum, int max_Intensity);
template <class myType>
void write_out_dicom_with_filter(std::string outputFileName, myType filter);
int threshold_with_itk_otsu(int numberOfThresholds, ReaderType::Pointer reader);
template <class myType>
int otsu_on_custom_threshold(int max_Intensity, int bins, myType reader);
template <class myType, class otherType>
int registration_image(myType fixedImageReader, otherType movingImageReader/*int argc, char *argv[]*/);
template <class myType>
VotBinFillHoleFilterType::Pointer segment_bone(myType reader, char* argv[], std::string a_out_folder_name);


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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " InputDicomDirectory  OutputDicomDirectory NumOfBins" << std::endl;
        return EXIT_FAILURE;
    }

    // Read DICOM
    NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    ReaderType::Pointer reader = ReaderType::New();

    namesGenerator->SetInputDirectory(argv[1]);

    const ReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();
    std::size_t numberOfFileNames = filenames.size();

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

    using MovingImageType = itk::Image< PixelType, Dimension >;
    using MovingImageReaderType = itk::ImageFileReader< MovingImageType >;
    MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
    
    movingImageReader->SetFileName("verse112.nii.gz");
    // movingImageReader->SetFileName("verse112_seg_bin.nii.gz");
    //movingImageReader->SetFileName("verse112_seg_bin_open.nii.gz");
    try
    {
        //fixedImageReader->Update();
        movingImageReader->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return -9;
    }
    /*
    using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(5);
    structuringElement.CreateStructuringElement();


    using BinaryMorphologicalClosingImageFilterType = itk::BinaryMorphologicalClosingImageFilter <ImageType, ImageType, StructuringElementType>;
    BinaryMorphologicalClosingImageFilterType::Pointer openingFilter = BinaryMorphologicalClosingImageFilterType::New();
    openingFilter->SetInput(movingImageReader->GetOutput());
    //openingFilter->SetBackgroundValue(0);
    openingFilter->SetForegroundValue(1);
    openingFilter->SetKernel(structuringElement);

    openingFilter->Update();

    write_out_dicom_with_filter<BinaryMorphologicalClosingImageFilterType::Pointer>("verse112_seg_bin_open", openingFilter);
    */
    VotBinFillHoleFilterType::Pointer fix = segment_bone<ReaderType::Pointer>(reader, argv, "FIX_IMAGE");
    VotBinFillHoleFilterType::Pointer mov = segment_bone<MovingImageReaderType::Pointer>(movingImageReader, argv, "MOV_IMAGE");
    //int result = registration_image<VotBinFillHoleFilterType::Pointer, MovingImageReaderType::Pointer>(fix, movingImageReader);
    int result = registration_image<VotBinFillHoleFilterType::Pointer, VotBinFillHoleFilterType::Pointer>(fix, mov);
    //int result = registration_image<ReaderType::Pointer>(reader);
    std::cout << "RESULT OF THE SEG: " << result << std::endl;
    /*
    // Use itk's algorithm
    t1 = std::chrono::high_resolution_clock::now();
    int threshold_itk = threshold_with_itk_otsu(4, reader);
    t2 = std::chrono::high_resolution_clock::now();
    auto duration_itk = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "itk threshold: " << threshold_itk << " time(ms): " << duration_itk << std::endl;

    std::cout << "ratio: " << (double(duration_itk) / double(duration_my)) << std::endl;

    //Thresholding
    //BinThFilterType::Pointer filter = BinThFilterType::New();
    th_filter->SetInput(reader->GetOutput());
    th_filter->SetLowerThreshold(threshold_itk);
    th_filter->SetUpperThreshold(9000);
    th_filter->SetOutsideValue(0);
    th_filter->SetInsideValue(4000);

    out_folder_name = argv[2];
    out_folder_name.append("_itk");
    write_out_dicom_with_filter<BinThFilterType::Pointer>(out_folder_name, th_filter, namesGenerator, gdcmIO, reader);

    // Fill hole
    //VotBinFillHoleFilterType::InputSizeType radius;
    radius.Fill(2);

    //VotBinFillHoleFilterType::Pointer filter = VotBinFillHoleFilterType::New();
    fill_hole_filter->SetInput(th_filter->GetOutput());
    fill_hole_filter->SetRadius(radius);
    fill_hole_filter->SetMajorityThreshold(1); // default is 1
    fill_hole_filter->SetBackgroundValue(0);
    fill_hole_filter->SetForegroundValue(4000);
    fill_hole_filter->SetMaximumNumberOfIterations(numberOfIterations);

    t1 = std::chrono::high_resolution_clock::now();
    //fill_hole_filter->Update();
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Fill hole time: " << duration << std::endl;

    out_folder_name.append("_fill_hole");
    //write_out_dicom_with_filter<VotBinFillHoleFilterType::Pointer>(out_folder_name, fill_hole_filter, namesGenerator, gdcmIO, reader);
*/
    return EXIT_SUCCESS;

}

template <class myType>
VotBinFillHoleFilterType::Pointer segment_bone(myType reader, char* argv[], std::string a_out_folder_name) {
    // for measure time
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;

    int numberOfIterations = 1;

    // Use my solution for otsu
    t1 = std::chrono::high_resolution_clock::now();
    int threshold_my = otsu_on_custom_threshold<myType>(std::atoi(argv[3]), std::atoi(argv[4]), reader);
    t2 = std::chrono::high_resolution_clock::now();
    auto duration_my = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "my threshold:  " << threshold_my << " time(ms): " << duration_my << std::endl;
    // Thresholding
    BinThFilterType::Pointer th_filter = BinThFilterType::New();
    th_filter->SetInput(reader->GetOutput());
    th_filter->SetLowerThreshold(threshold_my);
    th_filter->SetUpperThreshold(9000);
    th_filter->SetOutsideValue(0);
    th_filter->SetInsideValue(1);
    //std::string out_folder_name = argv[2];
    std::string out_folder_name = a_out_folder_name;
    out_folder_name.append("_custom");
    write_out_dicom_with_filter<BinThFilterType::Pointer>(out_folder_name, th_filter);


    std::cout << "Open radius: " << argv[5] << std::endl;
    // Open
    using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(std::atof(argv[5]));
    structuringElement.CreateStructuringElement();


    using BinaryMorphologicalOpeningImageFilterType = itk::BinaryMorphologicalOpeningImageFilter <ImageType, ImageType, StructuringElementType>;
    BinaryMorphologicalOpeningImageFilterType::Pointer openingFilter = BinaryMorphologicalOpeningImageFilterType::New();
    openingFilter->SetInput(th_filter->GetOutput());
    openingFilter->SetBackgroundValue(0);
    openingFilter->SetForegroundValue(1);
    openingFilter->SetKernel(structuringElement);

    t1 = std::chrono::high_resolution_clock::now();
    openingFilter->Update();
    t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Open time: " << duration << std::endl;

    out_folder_name.append("_open");
    write_out_dicom_with_filter<BinaryMorphologicalOpeningImageFilterType::Pointer>(out_folder_name, openingFilter);

    std::cout << "Fill hole radius: " << argv[6] << std::endl;
    // Fill hole
    VotBinFillHoleFilterType::InputSizeType radius;
    radius.Fill(std::atoi(argv[6]));

    VotBinFillHoleFilterType::Pointer fill_hole_filter = VotBinFillHoleFilterType::New();
    fill_hole_filter->SetInput(openingFilter->GetOutput());
    fill_hole_filter->SetRadius(radius);
    fill_hole_filter->SetMajorityThreshold(1); // default is 1
    fill_hole_filter->SetBackgroundValue(0);
    fill_hole_filter->SetForegroundValue(1);
    fill_hole_filter->SetMaximumNumberOfIterations(2);

    t1 = std::chrono::high_resolution_clock::now();
    fill_hole_filter->Update();
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Fill hole time: " << duration << std::endl;

    out_folder_name.append("_fill_hole");
    write_out_dicom_with_filter<VotBinFillHoleFilterType::Pointer>(out_folder_name, fill_hole_filter);

    return fill_hole_filter;

}

template <class myType>
void write_out_dicom_with_filter(std::string outputFileName, myType filter) {
    /*itksys::SystemTools::MakeDirectory(outputDirectory);

    SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
    seriesWriter->SetFileName(outputDirectory);

    seriesWriter->SetInput(filter->GetOutput()); // set the image to write
    seriesWriter->SetImageIO(gdcmIO);

    namesGenerator->SetOutputDirectory(outputDirectory);
    seriesWriter->SetFileNames(namesGenerator->GetOutputFileNames());
    //seriesWriter->SetMetaDataDictionaryArray(reader->GetMetaDataDictionaryArray());

    try {
        seriesWriter->Update();
    }
    catch (itk::ExceptionObject & excp) {
        std::cerr << "Exception thrown while writing the series " << std::endl;
        std::cerr << excp << std::endl;

        //return EXIT_FAILURE;
    }
    */
    using OutputPixelType = signed short;
    using OutputImageType = itk::Image< OutputPixelType, ImageDimension >;
    using CastFilterType = itk::CastImageFilter<
        FixedImageType,
        OutputImageType >;
    using WriterType = itk::ImageFileWriter< OutputImageType >;
    WriterType::Pointer      writer = WriterType::New();
    CastFilterType::Pointer  caster = CastFilterType::New();
    writer->SetFileName(outputFileName + ".nii.gz");
    caster->SetInput(filter->GetOutput());
    writer->SetInput(caster->GetOutput());
    try
    {
        writer->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }
}

int threshold_with_itk_otsu(int numberOfThresholds, ReaderType::Pointer reader) {
    using FilterType = itk::OtsuMultipleThresholdsImageFilter <ImageType, ImageType>;
    FilterType::Pointer otsuFilter
        = FilterType::New();
    otsuFilter->SetInput(reader->GetOutput());
    otsuFilter->SetNumberOfThresholds(numberOfThresholds);
    otsuFilter->Update(); // To compute threshold

    FilterType::ThresholdVectorType thresholds = otsuFilter->GetThresholds();

    return thresholds[3];

}

template<class myType >
int otsu_on_custom_threshold(int max_Intensity, int bins, myType reader) {
    ImageType::IndexType index3d;
    ImageType::SizeType size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();


    //int bins = std::atoi(argv[3]);
    int min_Intensity = 0;
    //int max_Intensity = 2064; -86
    int index = 0;
    uli sum = 0;

    uli * histogram;
    histogram = new uli[bins]();

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

    write_for_plot<uli>(histogram, bins, "histogram.txt");
    int threshold = my_otsu(histogram, bins, size, sum, max_Intensity);

    return threshold;
}

int my_otsu(uli* histogram, int bins, ImageType::SizeType img_size, uli sum, int max_Intensity) {

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
	for (int k = 0; k < bins; k++) {
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
	for (int t = 1; t < bins; t++) {
		q1next = q1prev + probability[t + 1]; //q1next-q1(t+1)
		mu1next = (q1prev*mu1 + (t + 1)*(probability[t + 1])) / q1next;//set mu1(t+1)
		mu2next = (mu - q1next * mu1next) / (1 - q1next);//set mu2(t+1)
		betweenvariance = q1prev * (1 - q1prev)*((mu1 - mu2)*(mu1 - mu2));//calculate between group variance
																		//max between group variance is initially set to 0. Change the max between group variance, and change the optimized threshold to t if the current variance is > max.
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
        var[t] = betweenvariance;
	}
	  
	 write_for_plot<float>(var, bins, "var.txt");

    /*
	std::cout << "\n";
	std::cout << "Threshold: " << optimizedthresh;
	std::cout << "\n";
	std::cout << "\n";
	std::cout << "\n";
    */

    return optimizedthresh * (max_Intensity / bins);
	  
}

class CommandIterationUpdate : public itk::Command
{
public:
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);
protected:
    CommandIterationUpdate() = default;
public:
    using OptimizerType = itk::RegularStepGradientDescentOptimizer;
    using OptimizerPointer = const OptimizerType *;
    void Execute(itk::Object *caller, const itk::EventObject & event) override
    {
        Execute((const itk::Object *)caller, event);
    }
    void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
        auto optimizer = static_cast< OptimizerPointer >(object);
        if (!(itk::IterationEvent().CheckEvent(&event)))
        {
            return;
        }
        std::cout << optimizer->GetCurrentIteration() << "   ";
        std::cout << optimizer->GetValue() << "   ";
        std::cout << std::endl;
    }
};

constexpr unsigned int ImageDimension = 3;
using PixelType = signed short;
using FixedImageType = itk::Image< PixelType, ImageDimension >;
using MovingImageType = itk::Image< PixelType, ImageDimension >;
using FixedImageReaderType = itk::ImageFileReader< FixedImageType  >;
using MovingImageReaderType = itk::ImageFileReader< MovingImageType >;

template <class myType, class otherType>
int registration_image(myType fixedImageReader, otherType movingImageReader/*int argc, char *argv[]*/)
{
    //string outputImagefile = argv[3];
    std::string outputImagefile = "REGIMAGE.nii.gz";
    char* movingImagefilePath = "verse112_seg_bin.nii.gz";
    //char* fixedImagefilePath = "verse112.nii.gz";
    int argc = 4;
    if (argc < 4)
    {
        std::cerr << "Missing Parameters " << std::endl;
        // std::cerr << "Usage: " << argv[0];
        std::cerr << " fixedImageFile  movingImageFile outputImagefile  ";
        std::cerr << " [differenceOutputfile] [differenceBeforeRegistration] ";
        std::cerr << " [filenameForFinalTransformParameters] ";
        std::cerr << " [useExplicitPDFderivatives ] [useCachingBSplineWeights ] ";
        std::cerr << " [deformationField] ";
        std::cerr << " [numberOfGridNodesInsideImageInOneDimensionCoarse] ";
        std::cerr << " [numberOfGridNodesInsideImageInOneDimensionFine] ";
        std::cerr << " [maximumStepLength] [maximumNumberOfIterations]";
        std::cerr << std::endl;
        return -10;
    }

    const unsigned int SpaceDimension = ImageDimension;
    constexpr unsigned int SplineOrder = 3;
    using CoordinateRepType = double;
    using RigidTransformType = itk::VersorRigid3DTransform< double >;
    using AffineTransformType = itk::AffineTransform< double, SpaceDimension >;
    using DeformableTransformType = itk::BSplineTransform<
        CoordinateRepType,
        SpaceDimension,
        SplineOrder >;
    using TransformInitializerType = itk::CenteredTransformInitializer<
        RigidTransformType,
        FixedImageType, MovingImageType >;
    using OptimizerType = itk::RegularStepGradientDescentOptimizer;
    using MetricType = itk::MattesMutualInformationImageToImageMetric<
        FixedImageType,
        MovingImageType >;
    using InterpolatorType = itk::LinearInterpolateImageFunction<
        MovingImageType,
        double          >;
    using RegistrationType = itk::ImageRegistrationMethod<
        FixedImageType,
        MovingImageType >;
    MetricType::Pointer         metric = MetricType::New();
    OptimizerType::Pointer      optimizer = OptimizerType::New();
    InterpolatorType::Pointer   interpolator = InterpolatorType::New();
    RegistrationType::Pointer   registration = RegistrationType::New();
    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    registration->SetInterpolator(interpolator);
    // Auxiliary identity transform.
    using IdentityTransformType = itk::IdentityTransform<double, SpaceDimension>;
    IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();
    //
    //   Read the Fixed and Moving images.
    //
    
    using FixedImageReaderType = itk::ImageFileReader< FixedImageType  >;
    //using MovingImageReaderType = itk::ImageFileReader< MovingImageType >;
    //MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
    ////comment this for custom image
    ////FixedImageReaderType::Pointer  fixedImageReader = FixedImageReaderType::New();
    ////fixedImageReader->SetFileName(fixedImagefilePath);
    //
    //movingImageReader->SetFileName(movingImagefilePath);
    //try
    //{
    ////fixedImageReader->Update();
    //movingImageReader->Update();
    //}
    //catch (itk::ExceptionObject & err)
    //{
    //std::cerr << "ExceptionObject caught !" << std::endl;
    //std::cerr << err << std::endl;
    //return -9;
    //}
    
    FixedImageType::ConstPointer fixedImage = fixedImageReader->GetOutput();
    registration->SetFixedImage(fixedImage);
    registration->SetMovingImage(movingImageReader->GetOutput());
    //
    // Add a time and memory probes collector for profiling the computation time
    // of every stage.
    //
    itk::TimeProbesCollectorBase chronometer;
    itk::MemoryProbesCollectorBase memorymeter;
    //
    // Setup the metric parameters
    //
    metric->SetNumberOfHistogramBins(50);
    FixedImageType::RegionType fixedRegion = fixedImage->GetBufferedRegion();
    const unsigned int numberOfPixels = fixedRegion.GetNumberOfPixels();
    //metric->ReinitializeSeed(76926294);
    if (argc > 7)
    {
        // Define whether to calculate the metric derivative by explicitly
        // computing the derivatives of the joint PDF with respect to the Transform
        // parameters, or doing it by progressively accumulating contributions from
        // each bin in the joint PDF.
        //metric->SetUseExplicitPDFDerivatives(std::stoi(argv[7]));
    }
    if (argc > 8)
    {
        // Define whether to cache the BSpline weights and indexes corresponding to
        // each one of the samples used to compute the metric. Enabling caching will
        // make the algorithm run faster but it will have a cost on the amount of memory
        // that needs to be allocated. This option is only relevant when using the
        // BSplineTransform.
        //metric->SetUseCachingOfBSplineWeights(std::stoi(argv[8]));
    }
    //
    //  Initialize a rigid transform by using Image Intensity Moments
    //
    TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    RigidTransformType::Pointer  rigidTransform = RigidTransformType::New();
    initializer->SetTransform(rigidTransform);
    initializer->SetFixedImage(fixedImageReader->GetOutput());
    initializer->SetMovingImage(movingImageReader->GetOutput());
    initializer->MomentsOn();
    std::cout << "Starting Rigid Transform Initialization " << std::endl;
    memorymeter.Start("Rigid Initialization");
    chronometer.Start("Rigid Initialization");
    initializer->InitializeTransform();
    chronometer.Stop("Rigid Initialization");
    memorymeter.Stop("Rigid Initialization");
    std::cout << "Rigid Transform Initialization completed" << std::endl;
    std::cout << std::endl;
    registration->SetFixedImageRegion(fixedRegion);
    registration->SetInitialTransformParameters(rigidTransform->GetParameters());
    registration->SetTransform(rigidTransform);
    //
    //  Define optimizer normaliztion to compensate for different dynamic range
    //  of rotations and translations.
    //
    using OptimizerScalesType = OptimizerType::ScalesType;
    OptimizerScalesType optimizerScales(rigidTransform->GetNumberOfParameters());
    const double translationScale = 1.0 / 1000.0;
    optimizerScales[0] = 1.0;
    optimizerScales[1] = 1.0;
    optimizerScales[2] = 1.0;
    optimizerScales[3] = translationScale;
    optimizerScales[4] = translationScale;
    optimizerScales[5] = translationScale;
    optimizer->SetScales(optimizerScales);
    optimizer->SetMaximumStepLength(0.1000);
    optimizer->SetMinimumStepLength(0.00000001);
    optimizer->SetNumberOfIterations(200);

    optimizer->SetRelaxationFactor(0.85);
    

    std::cout << "IS MIN: " << optimizer->GetMinimize() << "\n";

    //
    // The rigid transform has 6 parameters we use therefore a few samples to run
    // this stage.
    //
    // Regulating the number of samples in the Metric is equivalent to performing
    // multi-resolution registration because it is indeed a sub-sampling of the
    // image.
    metric->SetNumberOfSpatialSamples(10000L);
    //
    // Create the Command observer and register it with the optimizer.
    //
    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);
    std::cout << "Starting Rigid Registration " << std::endl;
    try
    {
        memorymeter.Start("Rigid Registration");
        chronometer.Start("Rigid Registration");
        registration->Update();
        chronometer.Stop("Rigid Registration");
        memorymeter.Stop("Rigid Registration");
        std::cout << "Optimizer stop condition = "
            << registration->GetOptimizer()->GetStopConditionDescription()
            << std::endl;
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return -8;
    }
    std::cout << "Rigid Registration completed" << std::endl;
    std::cout << std::endl;
    rigidTransform->SetParameters(registration->GetLastTransformParameters());
    //
    //  Perform Affine Registration
    //
    AffineTransformType::Pointer  affineTransform = AffineTransformType::New();
    affineTransform->SetCenter(rigidTransform->GetCenter());
    affineTransform->SetTranslation(rigidTransform->GetTranslation());
    affineTransform->SetMatrix(rigidTransform->GetMatrix());
    registration->SetTransform(affineTransform);
    registration->SetInitialTransformParameters(affineTransform->GetParameters());
    optimizerScales = OptimizerScalesType(affineTransform->GetNumberOfParameters());
    optimizerScales[0] = 1.0;
    optimizerScales[1] = 1.0;
    optimizerScales[2] = 1.0;
    optimizerScales[3] = 1.0;
    optimizerScales[4] = 1.0;
    optimizerScales[5] = 1.0;
    optimizerScales[6] = 1.0;
    optimizerScales[7] = 1.0;
    optimizerScales[8] = 1.0;
    optimizerScales[9] = translationScale;
    optimizerScales[10] = translationScale;
    optimizerScales[11] = translationScale;
    optimizer->SetScales(optimizerScales);
    optimizer->SetMaximumStepLength(0.3);
    optimizer->SetMinimumStepLength(0.00001);
    optimizer->SetNumberOfIterations(9000);

    optimizer->SetRelaxationFactor(0.5);
    optimizer->SetGradientMagnitudeTolerance(0.0000000001);
    //
    // The Affine transform has 12 parameters we use therefore a more samples to run
    // this stage.
    //
    // Regulating the number of samples in the Metric is equivalent to performing
    // multi-resolution registration because it is indeed a sub-sampling of the
    // image.
    metric->SetNumberOfSpatialSamples(50000L);
    std::cout << "Starting Affine Registration " << std::endl;
    try
    {
        memorymeter.Start("Affine Registration");
        chronometer.Start("Affine Registration");
        registration->Update();
        chronometer.Stop("Affine Registration");
        memorymeter.Stop("Affine Registration");
        std::cout << "Optimizer stop condition = "
            << registration->GetOptimizer()->GetStopConditionDescription()
            << std::endl;
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return -7;
    }
    std::cout << "Affine Registration completed" << std::endl;
    std::cout << std::endl;
    affineTransform->SetParameters(registration->GetLastTransformParameters());
    //
    //  Perform Deformable Registration
    //
    /*
    START ERROR: Joint PDF summed to zero
    DeformableTransformType::Pointer  bsplineTransformCoarse = DeformableTransformType::New();
    unsigned int numberOfGridNodesInOneDimensionCoarse = 5;
    DeformableTransformType::PhysicalDimensionsType   fixedPhysicalDimensions;
    DeformableTransformType::MeshSizeType             meshSize;
    DeformableTransformType::OriginType               fixedOrigin;
    for (unsigned int i = 0; i< SpaceDimension; i++)
    {
        fixedOrigin[i] = fixedImage->GetOrigin()[i];
        fixedPhysicalDimensions[i] = fixedImage->GetSpacing()[i] *
            static_cast<double>(
                fixedImage->GetLargestPossibleRegion().GetSize()[i] - 1);
    }
    meshSize.Fill(numberOfGridNodesInOneDimensionCoarse - SplineOrder);
    bsplineTransformCoarse->SetTransformDomainOrigin(fixedOrigin);
    bsplineTransformCoarse->SetTransformDomainPhysicalDimensions(
        fixedPhysicalDimensions);
    bsplineTransformCoarse->SetTransformDomainMeshSize(meshSize);
    bsplineTransformCoarse->SetTransformDomainDirection(
        fixedImage->GetDirection());
    using ParametersType = DeformableTransformType::ParametersType;
    unsigned int numberOfBSplineParameters = bsplineTransformCoarse->GetNumberOfParameters();
    optimizerScales = OptimizerScalesType(numberOfBSplineParameters);
    optimizerScales.Fill(1.0);
    optimizer->SetScales(optimizerScales);
    ParametersType initialDeformableTransformParameters(numberOfBSplineParameters);
    initialDeformableTransformParameters.Fill(0.0);
    bsplineTransformCoarse->SetParameters(initialDeformableTransformParameters);
    registration->SetInitialTransformParameters(bsplineTransformCoarse->GetParameters());
    registration->SetTransform(bsplineTransformCoarse);
    // Software Guide : EndCodeSnippet
    //  Software Guide : BeginLatex
    //
    //  Next we set the parameters of the RegularStepGradientDescentOptimizer object.
    //
    //  Software Guide : EndLatex
    // Software Guide : BeginCodeSnippet
    optimizer->SetMaximumStepLength(10.0);
    optimizer->SetMinimumStepLength(0.01);
    optimizer->SetRelaxationFactor(0.7);
    optimizer->SetNumberOfIterations(50);
    // Software Guide : EndCodeSnippet
    // Optionally, get the step length from the command line arguments
    //if (argc > 11)
    //{
        //PARAM
        //optimizer->SetMaximumStepLength(std::stod(argv[12]));
    //}
    // Optionally, get the number of iterations from the command line arguments
    //if (argc > 12)
    //{
        //PARAM
        //optimizer->SetNumberOfIterations(std::stoi(argv[13]));
    //}
    //
    // The BSpline transform has a large number of parameters, we use therefore a
    // much larger number of samples to run this stage.
    //
    // Regulating the number of samples in the Metric is equivalent to performing
    // multi-resolution registration because it is indeed a sub-sampling of the
    // image.
    metric->SetNumberOfSpatialSamples(numberOfBSplineParameters * 100);
    std::cout << std::endl << "Starting Deformable Registration Coarse Grid" << std::endl;
    try
    {
        memorymeter.Start("Deformable Registration Coarse");
        chronometer.Start("Deformable Registration Coarse");
        registration->Update();
        chronometer.Stop("Deformable Registration Coarse");
        memorymeter.Stop("Deformable Registration Coarse");
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Deformable Registration Coarse Grid completed" << std::endl;
    std::cout << std::endl;
    OptimizerType::ParametersType finalParameters =
        registration->GetLastTransformParameters();
    bsplineTransformCoarse->SetParameters(finalParameters);
    //  Software Guide : BeginLatex
    //
    //  Once the registration has finished with the low resolution grid, we
    //  proceed to instantiate a higher resolution
    //  \code{BSplineTransform}.
    //
    //  Software Guide : EndLatex
    DeformableTransformType::Pointer  bsplineTransformFine = DeformableTransformType::New();
    unsigned int numberOfGridNodesInOneDimensionFine = 20;
    meshSize.Fill(numberOfGridNodesInOneDimensionFine - SplineOrder);
    bsplineTransformFine->SetTransformDomainOrigin(fixedOrigin);
    bsplineTransformFine->SetTransformDomainPhysicalDimensions(
        fixedPhysicalDimensions);
    bsplineTransformFine->SetTransformDomainMeshSize(meshSize);
    bsplineTransformFine->SetTransformDomainDirection(
        fixedImage->GetDirection());
    numberOfBSplineParameters = bsplineTransformFine->GetNumberOfParameters();
    ParametersType parametersHigh(numberOfBSplineParameters);
    parametersHigh.Fill(0.0);
    //  Software Guide : BeginLatex
    //
    //  Now we need to initialize the BSpline coefficients of the higher resolution
    //  transform. This is done by first computing the actual deformation field
    //  at the higher resolution from the lower resolution BSpline coefficients.
    //  Then a BSpline decomposition is done to obtain the BSpline coefficient of
    //  the higher resolution transform.
    //
    //  Software Guide : EndLatex
    unsigned int counter = 0;
    for (unsigned int k = 0; k < SpaceDimension; k++)
    {
        using ParametersImageType = DeformableTransformType::ImageType;
        using ResamplerType = itk::ResampleImageFilter<ParametersImageType, ParametersImageType>;
        ResamplerType::Pointer upsampler = ResamplerType::New();
        using FunctionType = itk::BSplineResampleImageFunction<ParametersImageType, double>;
        FunctionType::Pointer function = FunctionType::New();
        upsampler->SetInput(bsplineTransformCoarse->GetCoefficientImages()[k]);
        upsampler->SetInterpolator(function);
        upsampler->SetTransform(identityTransform);
        upsampler->SetSize(bsplineTransformFine->GetCoefficientImages()[k]->
            GetLargestPossibleRegion().GetSize());
        upsampler->SetOutputSpacing(bsplineTransformFine->GetCoefficientImages()[k]->
            GetSpacing());
        upsampler->SetOutputOrigin(bsplineTransformFine->GetCoefficientImages()[k]->
            GetOrigin());
        using DecompositionType =
            itk::BSplineDecompositionImageFilter<ParametersImageType, ParametersImageType>;
        DecompositionType::Pointer decomposition = DecompositionType::New();
        decomposition->SetSplineOrder(SplineOrder);
        decomposition->SetInput(upsampler->GetOutput());
        decomposition->Update();
        ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();
        // copy the coefficients into the parameter array
        using Iterator = itk::ImageRegionIterator<ParametersImageType>;
        Iterator it(newCoefficients, bsplineTransformFine->GetCoefficientImages()[k]->
            GetLargestPossibleRegion());
        while (!it.IsAtEnd())
        {
            parametersHigh[counter++] = it.Get();
            ++it;
        }
    }
    optimizerScales = OptimizerScalesType(numberOfBSplineParameters);
    optimizerScales.Fill(1.0);
    optimizer->SetScales(optimizerScales);
    bsplineTransformFine->SetParameters(parametersHigh);
    //  Software Guide : BeginLatex
    //
    //  We now pass the parameters of the high resolution transform as the initial
    //  parameters to be used in a second stage of the registration process.
    //
    //  Software Guide : EndLatex
    std::cout << "Starting Registration with high resolution transform" << std::endl;
    // Software Guide : BeginCodeSnippet
    registration->SetInitialTransformParameters(
        bsplineTransformFine->GetParameters());
    registration->SetTransform(bsplineTransformFine);
    //
    // The BSpline transform at fine scale has a very large number of parameters,
    // we use therefore a much larger number of samples to run this stage. In
    // this case, however, the number of transform parameters is closer to the
    // number of pixels in the image. Therefore we use the geometric mean of the
    // two numbers to ensure that the number of samples is larger than the number
    // of transform parameters and smaller than the number of samples.
    //
    // Regulating the number of samples in the Metric is equivalent to performing
    // multi-resolution registration because it is indeed a sub-sampling of the
    // image.
    const auto numberOfSamples = static_cast<unsigned long>(
        std::sqrt(static_cast<double>(numberOfBSplineParameters) *
            static_cast<double>(numberOfPixels)));
    metric->SetNumberOfSpatialSamples(numberOfSamples);
    try
    {
        memorymeter.Start("Deformable Registration Fine");
        chronometer.Start("Deformable Registration Fine");
        registration->Update();
        chronometer.Stop("Deformable Registration Fine");
        memorymeter.Stop("Deformable Registration Fine");
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    // Software Guide : EndCodeSnippet
    std::cout << "Deformable Registration Fine Grid completed" << std::endl;
    std::cout << std::endl;

    finalParameters = registration->GetLastTransformParameters();
    bsplineTransformFine->SetParameters(finalParameters);
    END ERROR: Joint PDF summed to zero
    */

    using ResampleFilterType = itk::ResampleImageFilter<
        MovingImageType,
        FixedImageType >;
    ResampleFilterType::Pointer resample = ResampleFilterType::New();
    
    //resample->SetTransform(rigidTransform);
    resample->SetTransform(affineTransform);
    //resample->SetTransform(bsplineTransformCoarse);
    //resample->SetTransform(bsplineTransformFine);

    resample->SetInput(movingImageReader->GetOutput());
    
    resample->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resample->SetOutputOrigin(fixedImage->GetOrigin());
    resample->SetOutputSpacing(fixedImage->GetSpacing());
    resample->SetOutputDirection(fixedImage->GetDirection());
    // This value is set to zero in order to make easier to perform
    // regression testing in this example. However, for didactic
    // exercise it will be better to set it to a medium gray value
    // such as 100 or 128.
    resample->SetDefaultPixelValue(0);
    using OutputPixelType = signed short;
    using OutputImageType = itk::Image< OutputPixelType, ImageDimension >;
    using CastFilterType = itk::CastImageFilter<
        FixedImageType,
        OutputImageType >;
    using WriterType = itk::ImageFileWriter< OutputImageType >;
    WriterType::Pointer      writer = WriterType::New();
    CastFilterType::Pointer  caster = CastFilterType::New();
    writer->SetFileName(outputImagefile);
    caster->SetInput(resample->GetOutput());
    writer->SetInput(caster->GetOutput());
    std::cout << "Writing resampled moving image...";
    try
    {
        writer->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return -4;
    }
    std::cout << " Done!" << std::endl;
    return 0;
}