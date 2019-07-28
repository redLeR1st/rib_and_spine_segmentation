#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"

#include "itkMedianImageFilter.h"

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
#include <vector>

#include <iostream>
#include "getopt.h"
#include <cstdlib>

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
#include "itkImageSource.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"
#include "itkSqrtImageFilter.h"
#include "itkTransformFileWriter.h"

#include "itkCommand.h"

#include "itkAmoebaOptimizer.h"



const unsigned int Dimension = 3;

typedef signed short PixelType;
typedef signed short OutputPixelType;
const unsigned int OutputDimension = 2;


typedef itk::Image< PixelType, Dimension > ImageType;


typedef itk::Image< OutputPixelType, OutputDimension > Image2DType;

typedef itk::ImageSeriesReader< ImageType > ReaderDICOMType;
typedef itk::ImageFileReader< ImageType > ReaderType;
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
int threshold_with_itk_otsu(int numberOfThresholds, ReaderDICOMType::Pointer reader);
template <class myType>
int otsu_on_custom_threshold(int max_Intensity, int bins, myType reader);
template <class myType, class otherType, class thirdType>
int registration_image(myType fixedImageReader, otherType movingImageReader, thirdType resampleImageReader);
template <class myType>
VotBinFillHoleFilterType::Pointer segment_bone(myType reader, std::string a_out_folder_name);


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

static const char* input = NULL;
static const char* output = NULL;
static const char* max_Intensity = "2048";
static const char* num_of_bins = "128";
static const char* open_radius = "1";
static const char* fillhole_radius = "3";
static const char* moving_image = NULL;
static const char* do_resample_on_this = NULL;

/* Flag set by ‘--verbose’. */
static int verbose_flag;

static bool segment_on_fix_image = false;
static bool segment_on_mov_image = false;


int parse_cmd_arguments(int argc, char **argv) {

    int c;

    while (1) {
        static struct option long_options[] = {
            /* These options set a flag. */
            { "verbose", no_argument,       &verbose_flag, 1 },
        { "brief",   no_argument,       &verbose_flag, 0 },
        /* These options don’t set a flag.
        We distinguish them by their indices. */
        { "input",         required_argument, 0, 'i' },
        { "output",        required_argument, 0, 'o' },
        { "max-intensity", required_argument, 0, 'm' },
        { "bins",          required_argument, 0, 'b' },
        { "open",          required_argument, 0, 'p' },
        { "fillhole",      required_argument, 0, 'f' },
        { "moving",        required_argument, 0, 'v' },
        { "resample",      required_argument, 0, 'r' },

        { "segfix",        no_argument,       0, 's' },
        { "segmov",        no_argument,       0, 'e' },

        { "help",          no_argument,       0, 'h' },
        { 0, 0, 0, 0 }
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, "i:o:m:b:p:f:v:r:hse",
            long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c) {
        case 0:
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
                break;
            printf("option %s", long_options[option_index].name);
            if (optarg)
                printf(" with arg %s", optarg);
            printf("\n");
            break;

        case 'i':
            input = optarg;
            break;

        case 'o':
            output = optarg;
            break;

        case 'm':
            max_Intensity = optarg;
            break;

        case 'b':
            num_of_bins = optarg;
            break;

        case 'p':
            open_radius = optarg;
            break;

        case 'f':
            fillhole_radius = optarg;
            break;

        case 'v':
            moving_image = optarg;
            break;

        case 'r':
            do_resample_on_this = optarg;
            break;

        case 's':
            segment_on_fix_image = true;
            break;

        case 'e':
            segment_on_mov_image = true;
            break;

        case 'h':
            std::cerr << "Usage: " << argv[0]
                << "\n-i --input        Input, can be DICOM a directory, or nii.gz. file"
                << "\n-o --output       Output DICOM directory (out of use)."
                << "\n-m --max-intesity Max intesity on the histogram (must be dividable by -b -bins, ex: -m 2064 -b 86) (def: 2048)."
                << "\n-b --bins         Bins to use in historgramm. (def: 128)"
                << "\n-p --open         Open radius. (def: 1)"
                << "\n-f --fillhole     Fill hole radius. (def: 3)"
                << "\n-v --moving       Image to register to the input image."
                << "\n-r --resample     Adapt transformations on this image."
                << "\n-s --segfix       Segment the fix image."
                << "\n-e --segmov       Segment the moving image."
                << "\n-h --help         Prints this message."
                << "\n";
            break;

        case '?':
            /* getopt_long already printed an error message. */
            break;

        default:
            abort();
        }
    }

    /* Instead of reporting ‘--verbose’
    and ‘--brief’ as they are encountered,
    we report the final status resulting from them. */
    if (verbose_flag) {
        puts("verbose flag is set");
    }

    /* Print any remaining command line arguments (not options). */
    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc)
            printf("%s ", argv[optind++]);
        putchar('\n');
    }
}


int main(int argc, char* argv[]) {
    parse_cmd_arguments(argc, argv);



    // Read the moving and resample image
    using MovingImageType = itk::Image< PixelType, Dimension >;
    using MovingImageReaderType = itk::ImageFileReader< MovingImageType >;
    MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
    MovingImageReaderType::Pointer resampleImageReader = MovingImageReaderType::New();

    if (moving_image != NULL) {
        movingImageReader->SetFileName(moving_image);

    }
    else {
        std::cout << "Error: -v --moving Moving image must be set.\n";
        return EXIT_FAILURE;
    }
    if (do_resample_on_this == NULL) {
        resampleImageReader->SetFileName(moving_image);
    }
    else {
        resampleImageReader->SetFileName(do_resample_on_this);
    }

    try
    {
        movingImageReader->Update();
        resampleImageReader->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught!" << std::endl;
        std::cerr << err << std::endl;
    }

    int result = 1;

    if (input != NULL) {
        if (strstr(input, "nii.gz") != NULL) {
            // Read nii.gz
            ReaderType::Pointer reader = ReaderType::New();
            reader->SetFileName(input);

            // Segmentation and regsitration
            if (segment_on_fix_image) {
                VotBinFillHoleFilterType::Pointer fix = segment_bone<ReaderType::Pointer>(reader, "FIX_IMAGE");
                result = registration_image<VotBinFillHoleFilterType::Pointer, MovingImageReaderType::Pointer, MovingImageReaderType::Pointer>(fix, movingImageReader, resampleImageReader);
            }
            else if (segment_on_mov_image) {
                VotBinFillHoleFilterType::Pointer mov = segment_bone<MovingImageReaderType::Pointer>(movingImageReader, "MOV_IMAGE");
                result = registration_image<ReaderType::Pointer, VotBinFillHoleFilterType::Pointer, MovingImageReaderType::Pointer>(reader, mov, resampleImageReader);
            }
            else if (segment_on_fix_image && segment_on_mov_image) {
                VotBinFillHoleFilterType::Pointer fix = segment_bone<ReaderType::Pointer>(reader, "FIX_IMAGE");
                VotBinFillHoleFilterType::Pointer mov = segment_bone<MovingImageReaderType::Pointer>(movingImageReader, "MOV_IMAGE");
                result = registration_image<VotBinFillHoleFilterType::Pointer, VotBinFillHoleFilterType::Pointer, MovingImageReaderType::Pointer>(fix, mov, resampleImageReader);
            }
            else {
                result = registration_image<ReaderType::Pointer, MovingImageReaderType::Pointer, MovingImageReaderType::Pointer>(reader, movingImageReader, resampleImageReader);
            }

        }
        else {
            // Read DICOM
            NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
            ImageIOType::Pointer gdcmIO = ImageIOType::New();


            namesGenerator->SetInputDirectory(input);
            const ReaderDICOMType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();
            std::size_t numberOfFileNames = filenames.size();
            ReaderDICOMType::Pointer reader = ReaderDICOMType::New();
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

            // Segmentation and regsitration
            if (segment_on_fix_image) {
                VotBinFillHoleFilterType::Pointer fix = segment_bone<ReaderDICOMType::Pointer>(reader, "FIX_IMAGE");
                result = registration_image<VotBinFillHoleFilterType::Pointer, MovingImageReaderType::Pointer, MovingImageReaderType::Pointer>(fix, movingImageReader, resampleImageReader);
            }
            else if (segment_on_mov_image) {
                VotBinFillHoleFilterType::Pointer mov = segment_bone<MovingImageReaderType::Pointer>(movingImageReader, "MOV_IMAGE");
                result = registration_image<ReaderDICOMType::Pointer, VotBinFillHoleFilterType::Pointer, MovingImageReaderType::Pointer>(reader, mov, resampleImageReader);
            }
            else if (segment_on_fix_image && segment_on_mov_image) {
                VotBinFillHoleFilterType::Pointer fix = segment_bone<ReaderDICOMType::Pointer>(reader, "FIX_IMAGE");
                VotBinFillHoleFilterType::Pointer mov = segment_bone<MovingImageReaderType::Pointer>(movingImageReader, "MOV_IMAGE");
                result = registration_image<VotBinFillHoleFilterType::Pointer, VotBinFillHoleFilterType::Pointer, MovingImageReaderType::Pointer>(fix, mov, resampleImageReader);
            }
            else {
                result = registration_image<ReaderDICOMType::Pointer, MovingImageReaderType::Pointer, MovingImageReaderType::Pointer>(reader, movingImageReader, resampleImageReader);
            }
        }
    }
    else {
        std::cout << "Error: -i --input Input DICOM directory must be set.\n";
        return EXIT_FAILURE;
    }

    std::cout << "RESULT OF THE SEG: " << result << std::endl;
    return EXIT_SUCCESS;

}

template <class myType>
VotBinFillHoleFilterType::Pointer segment_bone(myType reader, std::string a_out_folder_name) {
    // for measure time
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;

    int numberOfIterations = 1;

    // Use my solution for otsu
    t1 = std::chrono::high_resolution_clock::now();
    int threshold_my = otsu_on_custom_threshold<myType>(std::atoi(max_Intensity), std::atoi(num_of_bins), reader);
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
    //std::string out_folder_name = output;
    std::string out_folder_name = a_out_folder_name;
    out_folder_name.append("_custom");
    write_out_dicom_with_filter<BinThFilterType::Pointer>(out_folder_name, th_filter);


    std::cout << "Open radius: " << open_radius << std::endl;
    // Open
    using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(std::atof(open_radius));
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

    std::cout << "Fill hole radius: " << fillhole_radius << std::endl;
    // Fill hole
    VotBinFillHoleFilterType::InputSizeType radius;
    radius.Fill(std::atoi(fillhole_radius));

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

int threshold_with_itk_otsu(int numberOfThresholds, ReaderDICOMType::Pointer reader) {
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


    int min_Intensity = 0;
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
    probability = new float[bins]();

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
    using OptimizerType = itk::AmoebaOptimizer;
    using OptimizerPointer = const OptimizerType   *;
    void Execute(itk::Object *caller, const itk::EventObject & event) override
    {
        Execute((const itk::Object *)caller, event);
    }
    void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
        auto optimizer = static_cast< OptimizerPointer >(object);
        if (!itk::IterationEvent().CheckEvent(&event))
        {
            return;
        }
        std::cout << optimizer->GetCachedValue() << "   ";
        std::cout << optimizer->GetCachedCurrentPosition() << std::endl;
    }
};

constexpr unsigned int ImageDimension = 3;
using PixelType = signed short;
using FixedImageType = itk::Image< PixelType, ImageDimension >;
using MovingImageType = itk::Image< PixelType, ImageDimension >;
using FixedImageReaderType = itk::ImageFileReader< FixedImageType  >;
using MovingImageReaderType = itk::ImageFileReader< MovingImageType >;

template <class myType, class otherType, class thirdType>
int registration_image(myType fixedImageReader, otherType movingImageReader, thirdType resampleImageReader)
{
    std::string outputImagefile = "REGIMAGE.nii.gz";

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

    //using OptimizerType = itk::RegularStepGradientDescentOptimizer;
    using OptimizerType = itk::AmoebaOptimizer;

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

    /*
    optimizer->SetParametersConvergenceTolerance(0.25); // quarter pixel
    optimizer->SetFunctionConvergenceTolerance(0.001);
    */

    optimizer->SetOptimizeWithRestarts(true);

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

    /*
    optimizer->SetParametersConvergenceTolerance(0.25); // quarter pixel
    optimizer->SetFunctionConvergenceTolerance(0.001);
    */

    optimizer->SetOptimizeWithRestarts(true);

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

    //START ERROR: Joint PDF summed to zero
    /*
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
    // END ERROR: Joint PDF summed to zero
    */

    using ResampleFilterType = itk::ResampleImageFilter<
        MovingImageType,
        FixedImageType >;
    ResampleFilterType::Pointer resample = ResampleFilterType::New();

    //resample->SetTransform(rigidTransform);
    resample->SetTransform(affineTransform);
    //resample->SetTransform(bsplineTransformCoarse);
    //resample->SetTransform(bsplineTransformFine);

    //resample->SetInput(movingImageReader->GetOutput());
    resample->SetInput(resampleImageReader->GetOutput());

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