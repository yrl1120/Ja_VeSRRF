
import java.awt.image.DataBuffer;
import java.awt.image.BufferedImage;

import java.awt.image.ColorModel;
import java.awt.image.RenderedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Iterator;
import javax.imageio.*;
import javax.imageio.stream.ImageInputStream;
import javax.imageio.stream.ImageOutputStream;
import javax.imageio.ImageIO;
import java.util.List;
import java.util.ArrayList;
import java.awt.image.DataBufferUShort;
import javax.swing.JFileChooser;

import ij.*;
import ij.process.*;
import ij.plugin.*;


public class VRRF_ implements PlugIn {
    public void run(String arg) {

        //public static void main(String[] args) {
        ImagePlus imp = IJ.openImage();
        //IJ.run("eSRRF - Parameters sweep", "show calculate_0 sigma=1 crop pixels=3 calculate_1 use processing=[Default device] analysis=20000 enable magnification=6 avg wide-field start=1 delta=0.50 number=2 start_0=1 delta_0=1 number_0=1 start_1=1 delta_1=1 number_1=1 show_0");
        ImageProcessor ip = imp.getProcessor();
        int width = ip.getWidth();
        int height = ip.getHeight();

        int size = imp.getStackSize();

// 创建用于存储像素值的三维数组
        double[][][] pixelValues = new double[size][height][width];

// 遍历每一帧的图像
        for (int frame = 0; frame < size; frame++) {
            // 设置当前帧
            imp.setSlice(frame + 1);

            // 获取当前帧的 ImageProcessor 对象
            ImageProcessor frameIp = imp.getProcessor();

            // 遍历图像的每个像素
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // 获取像素值
                    double pixelValue = frameIp.getPixel(x, y);

                    // 存储像素值到数组中
                    pixelValues[frame][y][x] = pixelValue;
                }
            }
        }



        //String readPath = "C:\\Users\\cza42344\\JAVA\\ImageProcessing\\src\\ImageProcessing\\";
        //String savePath = "C:\\Users\\cza42344\\JAVA\\ImageProcessing\\src\\ImageProcessing\\OUT\\";
        //String readTitle = "100.tif";
        //String saveTitle = "output_image.tif";

        //File inputFile = new File(readPath + readTitle);
        //BufferedImage[] images = readImageStack(inputFile,500);


        //int width = images[0].getWidth();
        //int height = images[0].getHeight();
        //int size = images.length;

        double[][][] Io = new double[size][height][width];

        for (int i = 0; i < size; i++) {
            //BufferedImage I1 = images[i];
            //Io[i] = convertTo2DGrayscale(images[i]);
            Io[i] = subtractMinValue(pixelValues[i]);

        }

        // Interpolation
        int mag = 2;
        double[][][] I1 = new double[size][height*2][width*2];
        I1 = resizeImageStack(Io, mag);



        double[][][] gY = gradientX(I1);
        double[][][] gX = gradientY(I1);

        double[][][] gradients = calculateGradients(I1);
        double[][] Gy_ave = averageGradients(gY);
        double[][] Gx_ave = averageGradients(gX);
        double[][] Gy_var = calculateVariance(gY, Gy_ave);
        double[][] Gx_var = calculateVariance(gX, Gx_ave);
        double[][] Gr_var = calculateTotalGradientVariance(Gy_var, Gx_var);

        double[][] Io_ave = averageImageStack(I1);
        double[][] Io_var = calculateImageVariance(I1, Io_ave);

        // Thresholding
        double th = 0.1 * getMaxValue(Io_var);


        // Reweighting
        double[][] W = new double[height*2][width*2];
        double maxIvar = getMaxValue(Io_var);
        double maxGvar = getMaxValue(Gr_var);
        Gr_var = normalizeImage(Gr_var, maxGvar);


        double[][][] updatedI = applyThresholding(I1, Io_var, Gr_var, th, maxIvar);


        //short[][][] convertedImage = convertTo16Bit(updatedI);
        int[][][] converted = new int[size][height*2][width*2];
        for (int s = 0; s < size; s++) {
            for (int h = 0; h < height*2; h++) {
                for (int w = 0; w < width*2; w++) {
                    converted[s][h][w] = (int) Math.round( updatedI[s][h][w]);
                }
            }
        }

        double[][][] I = new double[size][height*2][width*2];

        for (int s = 0; s < size; s++) {
            I[s] = multiplyImages(updatedI[s]);
        }


        ImageStack stack = new ImageStack(width*2, height*2);

        // 将每帧的像素值添加到ImageStack对象中
        for (int frame = 0; frame < size; frame++) {
            double[][] framePixels = I[frame];

            // 创建FloatProcessor对象
            FloatProcessor fp = new FloatProcessor(width*2, height*2);

            // 将像素值设置到FloatProcessor对象中
            for (int y = 0; y < height*2; y++) {
                for (int x = 0; x < width*2; x++) {
                    float pixelValue =(float) framePixels[y][x];
                    fp.setf(x, y, pixelValue);
                }
            }

            // 将FloatProcessor对象添加到ImageStack中
            stack.addSlice(fp);
        }
        ImagePlus im = new ImagePlus("My Image Sequence", stack);
        IJ.log("Select the path to save the image sequence after VRRF processing");
        //String savePath = "C:\\Users\\cza42344\\Desktop\\test\\OUT\\ ";
        //IJ.saveAs(im, "Tiff",savePath);
        JFileChooser fileChooser = new JFileChooser();

        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        int option = fileChooser.showSaveDialog(null);
        if (option == JFileChooser.APPROVE_OPTION) {
            String saveFolder = fileChooser.getSelectedFile().getAbsolutePath();

            String savePath = saveFolder + "\\VRRF_result.tif";
            IJ.saveAs(im, "Tiff", savePath);
        }



        // Save modified image stack
/*
//        String saveFilePath = savePath + saveTitle;
        try {
            writeImageStack(I, savePath,saveTitle);
        } catch (IOException e) {
            e.printStackTrace();
        }
*/

    }


    public static void main(String[] args) {

        VRRF_ plugin = new VRRF_();
        plugin.run("");
    }

    public static double[][][] applyThresholding(double[][][] I, double[][] Ivar, double[][] Gvar, double th, double maxIvar) {
        int size = I.length;
        int height = I[0].length;
        int width = I[0][0].length;
        double[][][] updatedI = new double[size][height][width];
        double[][] W = new double[height][width];

        // Thresholding to ensure convergence in background areas
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (Ivar[i][j] >= th) {
                    W[i][j] = Ivar[i][j] / (Gvar[i][j] * ((Ivar[i][j] - th) / maxIvar) + 1);
                } else {
                    W[i][j] = Ivar[i][j];
                }
            }
        }

        // Normalize W
        double maxW = Double.MIN_VALUE;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (W[i][j] > maxW) {
                    maxW = W[i][j];
                }
            }
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                W[i][j] /= maxW;
            }
        }

        // Calculate bias
        double bias = th / maxIvar;

        // Update I
        for (int s = 0; s < size; s++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    updatedI[s][i][j] = I[s][i][j] * (bias + W[i][j]);
                }
            }
        }

        return updatedI;
    }

    private static double[][][] gradientX(double[][][] f) {

        int size = f.length;
        int width = f[0][0].length;
        int height = f[0].length;

        double[][][] g = new double[size][height][width];
        double[] h = new double[height];// Assuming you have the corresponding coordinates for the height dimension
        for (int i = 0; i < height; i++) {
            h[i] = i + 1;
        }
// Loop over each slice in the stack
        for (int s = 0; s < size; s++) {
            // Take forward differences on left and right edges
            if (height > 1) {
                for (int w = 0; w < width; w++) {
                    g[s][0][w] = (f[s][1][w] - f[s][0][w]) / (h[1] - h[0]);
                    g[s][height - 1][w] = (f[s][height - 1][w] - f[s][height - 2][w]) /(h[height - 1] - h[height - 2]);
                }
            }

            // Take centered differences on interior points
            if (height > 2) {
                for (int i = 1; i < height - 1; i++) {
                    for (int w = 0; w < width; w++) {
                        g[s][i][w] = (f[s][i + 1][w] - f[s][i - 1][w]) /(h[i + 1] - h[i - 1]);
                    }
                }
            }
        }



        return g;
    }


    private static double[][][] gradientY(double[][][] f) {

        int size = f.length;
        int width = f[0][0].length;
        int height = f[0].length;

        double[][][] g = new double[size][height][width];

        double[] h2 = new double[width]; // Assuming you have the corresponding coordinates for the width dimension
        for (int i = 0; i < width; i++) {
            h2[i] = i + 1;
        }

// Loop over each slice in the stack
        for (int s = 0; s < size; s++) {
            // Take forward differences on left and right edges
            if (width > 1) {
                for (int h = 0; h < height; h++) {
                    g[s][h][0] = (f[s][h][1] - f[s][h][0]) / (h2[1] - h2[0]);
                    g[s][h][width - 1] = (f[s][h][width - 1] - f[s][h][width - 2]) / (h2[width - 1] - h2[width - 2]);
                }
            }

            // Take centered differences on interior points
            if (width > 2) {
                for (int h = 0; h < height; h++) {
                    for (int i = 1; i < width - 1; i++) {
                        g[s][h][i] = (f[s][h][i + 1] - f[s][h][i - 1]) / (h2[i + 1] - h2[i - 1]);
                    }
                }
            }
        }

        return g;

    }
    private static BufferedImage[] readImageStack(File inputFile, int length) {
        BufferedImage[] images = null;
        try {
            ImageInputStream imageStream = ImageIO.createImageInputStream(inputFile);
            Iterator<ImageReader> readers = ImageIO.getImageReaders(imageStream);
            if (readers.hasNext()) {
                ImageReader reader = readers.next();
                reader.setInput(imageStream);

                // 如果 tiff 文件是多帧的，则只取前 length 个帧
                int numImages = reader.getNumImages(true);
                int size = Math.min(length, numImages);
                images = new BufferedImage[size];

                // 使用 JAI 转换 tiff 文件到 BufferedImage
                for (int i = 0; i < size; i++) {
                    images[i] = reader.read(i);
                    /*
                    ParameterBlock paramBlock = new ParameterBlock();
                    paramBlock.add(reader);
                    paramBlock.add(i);
                    RenderedImage image = JAI.create("fileload", inputFile.getAbsolutePath());
                    images[i] = convertRenderedImage(image);

                     */
                }

                reader.dispose();
                imageStream.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return images;
    }

    // 将 RenderedImage 转为 BufferedImage
    private static BufferedImage convertRenderedImage(RenderedImage img) {
        if (img instanceof BufferedImage) {
            return (BufferedImage) img;
        }

        ColorModel cm = img.getColorModel();
        int width = img.getWidth();
        int height = img.getHeight();
        WritableRaster raster = cm.createCompatibleWritableRaster(width, height);
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        Hashtable<String, Object> properties = new Hashtable<>();
        String[] keys = img.getPropertyNames();
        if (keys != null) {
            for (String key : keys) {
                properties.put(key, img.getProperty(key));
            }
        }

        BufferedImage result = new BufferedImage(cm, raster, isAlphaPremultiplied, properties);
        img.copyData(raster);
        return result;
    }

    private static double[][] convertTo2DGrayscale(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        //int[][] grayscaleImage = new int[height][width];
        double[][] pixels = new double[height][width];

        DataBuffer dataBuffer = image.getRaster().getDataBuffer();
        if (dataBuffer.getDataType() == DataBuffer.TYPE_USHORT) {
            // Process the pixels
            // ...
        } else {
            System.out.println("The image is not 16-bit grayscale.");
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 获取坐标 (x, y) 处的像素值
                int pixel = dataBuffer.getElem(y * width + x);
                pixels[y][x] = pixel;

            }
        }

        //return grayscaleImage;
        return pixels;
    }

    private static double[][] subtractMinValue(double[][] image) {
        int width = image[0].length;
        int height = image.length;
        double min = image[0][0];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (image[y][x] < min) {
                    min = image[y][x];
                }
            }
        }
        double[][] result = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result[y][x] = image[y][x] - min;
            }
        }
        return result;
    }

    private static double[][][] resizeImageStack(double[][][] imageStack, int scaleFactor) {
        int size = imageStack.length;
        int width = imageStack[0][0].length;
        int height = imageStack[0].length;
        int newWidth = width * scaleFactor;
        int newHeight = height * scaleFactor;
        double[][][] resizedImageStack = new double[size][newHeight][newWidth];


        for (int s = 0; s < size; s++) {
            double[][] inputImage = imageStack[s];
            double[][] outputImage = new double[newHeight][newWidth];
            double scaleX = (double) width / newWidth;
            double scaleY = (double) height / newHeight;



            double[][] weightHeight = new double[newHeight][6];
            int[][] indicesHeight = new int[newHeight][6];
            double[] xh = new double[newHeight];
            for (int i = 0; i < newHeight; i++) {
                xh[i] = i + 1;
            }
            double[] uh = new double[newHeight];
            for (int i = 0; i < newHeight; i++) {
                uh[i] = xh[i] / 2 + 0.5 * 0.5;
            }
            int[] left = new int[newHeight];
            double kernelWidth = 4.0;
            for (int i = 0; i < newHeight; i++) {
                left[i] = (int) Math.floor(uh[i] - kernelWidth / 2);
            }
            int P = (int) Math.ceil(kernelWidth) + 2;

            for (int i = 0; i < left.length; i++) {
                //indicesHeight[i] = new int[left.length];
                for (int j = 0; j < P; j++) {
                    indicesHeight[i][j] = left[i] + j;
                }
            }
            for (int i = 0; i < indicesHeight.length; i++) {
                for (int j = 0; j < indicesHeight[0].length; j++) {
                    double x = uh[i] - indicesHeight[i][j];
                    weightHeight[i][j] = cubic(x);
                }
            }
            for (int i = 0; i < weightHeight.length; i++) {
                double sum = 0;
                for (int j = 0; j < weightHeight[i].length; j++) {
                    sum += weightHeight[i][j];
                }
                for (int j = 0; j < weightHeight[i].length; j++) {
                    weightHeight[i][j] /= sum;
                }
            }
            int[][] mirroredIndices = new int[indicesHeight.length][indicesHeight[0].length];

            int[] aux = new int[newHeight];
            for (int i = 0; i < height; i++) {
                aux[i] = i + 1;
                aux[height + i] = height - i;
            }

            for (int i = 0; i < indicesHeight.length; i++) {
                for (int j = 0; j < indicesHeight[0].length; j++) {
                    int index = indicesHeight[i][j];
                    int modIndex = (index - 1) % aux.length;
                    if (modIndex < 0) {
                        modIndex += aux.length;
                    }
                    mirroredIndices[i][j] = aux[modIndex];
                    indicesHeight[i][j]= mirroredIndices[i][j];

                    //indicesHeight[i][j] = Math.max(0, Math.min(mirroredIndices[i][j], height));
                }
            }

            List<Integer> columnsToDelete = new ArrayList<>();

// 遍历每一列
            for (int j = 0; j < weightHeight[0].length; j++) {
                boolean allZeros = true;

                // 检查是否全为零
                for (int i = 0; i < weightHeight.length; i++) {
                    if (weightHeight[i][j] != 0) {
                        allZeros = false;
                        break;
                    }
                }

                // 如果全为零，将列索引添加到列表中
                if (allZeros) {
                    columnsToDelete.add(j);
                }
            }

// 删除全为零的列
            if (!columnsToDelete.isEmpty()) {
                // 从权重矩阵中删除列
                double[][] updatedWeights = new double[weightHeight.length][weightHeight[0].length - columnsToDelete.size()];
                for (int i = 0; i < weightHeight.length; i++) {
                    int destCol = 0;
                    for (int srcCol = 0; srcCol < weightHeight[0].length; srcCol++) {
                        if (!columnsToDelete.contains(srcCol)) {
                            updatedWeights[i][destCol++] = weightHeight[i][srcCol];
                        }
                    }
                }
                weightHeight = updatedWeights;

                // 从索引矩阵中删除列
                int[][] updatedIndices = new int[indicesHeight.length][indicesHeight[0].length - columnsToDelete.size()];
                for (int i = 0; i < indicesHeight.length; i++) {
                    int destCol = 0;
                    for (int srcCol = 0; srcCol < indicesHeight[0].length; srcCol++) {
                        if (!columnsToDelete.contains(srcCol)) {
                            updatedIndices[i][destCol++] = indicesHeight[i][srcCol];
                        }
                    }
                }
                indicesHeight = updatedIndices;

                // 更新数组的大小
                //int newSize = indices[0].length;
            }



            double[][] weightWidth = new double[newWidth][6];
            int[][] indicesWidth = new int[newWidth][6];
            double[] xw = new double[newWidth];
            for (int i = 0; i < newWidth; i++) {
                xw[i] = i + 1;
            }
            double[] uw = new double[newWidth];
            for (int i = 0; i < newWidth; i++) {
                uw[i] = xw[i] / 2 + 0.5 * 0.5;
            }
            int[] leftw = new int[newWidth];
            for (int i = 0; i < newWidth; i++) {
                leftw[i] = (int) Math.floor(uw[i] - kernelWidth / 2);
            }
            int Pw = (int) Math.ceil(kernelWidth) + 2;

            for (int i = 0; i < leftw.length; i++) {
                //indicesHeight[i] = new int[left.length];
                for (int j = 0; j < Pw; j++) {
                    indicesWidth[i][j] = leftw[i] + j;
                }
            }
            for (int i = 0; i < indicesWidth.length; i++) {
                for (int j = 0; j < indicesWidth[0].length; j++) {
                    double x = uw[i] - indicesWidth[i][j];
                    weightWidth[i][j] = cubic(x);
                }
            }
            for (int i = 0; i < weightHeight.length; i++) {
                double sum = 0;
                for (int j = 0; j < weightHeight[i].length; j++) {
                    sum += weightHeight[i][j];
                }
                for (int j = 0; j < weightHeight[i].length; j++) {
                    weightHeight[i][j] /= sum;
                }
            }
            int[][] mirroredIndicesw = new int[indicesWidth.length][indicesWidth[0].length];

            int[] auxw = new int[newWidth];
            for (int i = 0; i < width; i++) {
                auxw[i] = i + 1;
                auxw[width + i] = width - i;
            }

            for (int i = 0; i < indicesWidth.length; i++) {
                for (int j = 0; j < indicesWidth[0].length; j++) {
                    int index = indicesWidth[i][j];
                    int modIndex = (index - 1) % auxw.length;
                    if (modIndex < 0) {
                        modIndex += auxw.length;
                    }
                    mirroredIndicesw[i][j] = auxw[modIndex];
                    indicesWidth[i][j]= mirroredIndicesw[i][j];

                    //indicesWidth[i][j] = Math.max(0, Math.min(mirroredIndicesw[i][j], width));
                }
            }

            //removeZeroColumns(weightWidth, indicesWidth);
            List<Integer> columnsToDelete2 = new ArrayList<>();

// 遍历每一列
            for (int j = 0; j < weightWidth[0].length; j++) {
                boolean allZeros = true;

                // 检查是否全为零
                for (int i = 0; i < weightWidth.length; i++) {
                    if (weightWidth[i][j] != 0) {
                        allZeros = false;
                        break;
                    }
                }

                // 如果全为零，将列索引添加到列表中
                if (allZeros) {
                    columnsToDelete2.add(j);
                }
            }

// 删除全为零的列
            if (!columnsToDelete2.isEmpty()) {
                // 从权重矩阵中删除列
                double[][] updatedWeights = new double[weightWidth.length][weightWidth[0].length - columnsToDelete2.size()];
                for (int i = 0; i < weightWidth.length; i++) {
                    int destCol = 0;
                    for (int srcCol = 0; srcCol < weightWidth[0].length; srcCol++) {
                        if (!columnsToDelete2.contains(srcCol)) {
                            updatedWeights[i][destCol++] = weightWidth[i][srcCol];
                        }
                    }
                }
                weightWidth = updatedWeights;

                // 从索引矩阵中删除列
                int[][] updatedIndices = new int[indicesWidth.length][indicesWidth[0].length - columnsToDelete2.size()];
                for (int i = 0; i < indicesWidth.length; i++) {
                    int destCol = 0;
                    for (int srcCol = 0; srcCol < indicesWidth[0].length; srcCol++) {
                        if (!columnsToDelete2.contains(srcCol)) {
                            updatedIndices[i][destCol++] = indicesWidth[i][srcCol];
                        }
                    }
                }
                indicesWidth = updatedIndices;

                // 更新数组的大小
                // int newSize = v[0].length;
            }

            double[][][] interpolatedImageHeight = interpolateImage(imageStack, weightHeight, indicesHeight);
            double[][][] interpolatedImageWidth = interpolateImageWidth(interpolatedImageHeight, weightWidth, indicesWidth);

            resizedImageStack= interpolatedImageWidth;


        }

        return resizedImageStack;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
    public static double[][][] resizeAlongDim(double[][][] in, int dim, double[][] weights, int[][] indices) {
        int size = in.length;
        int height = in[0].length;
        int width = in[0][0].length;

        int outLength = weights.length;
        int newHeight = height * 2;
        double[][][] out = new double[size][newHeight][width];

        for (int k = 0; k < size; k++) {
            for (int i = 0; i < newHeight; i++) {
                for (int j = 0; j < width; j++) {
                    int index = indices[k][i];
                    out[k][i][j] = in[k][i][j] * weights[i][j];
                }
            }
        }

        return out;
    }
    public static double[][][] interpolateImage(double[][][] image, double[][] weights, int[][] indices) {
        int depth = image.length;
        int height = image[0].length;
        int width = image[0][0].length;
        int newHeight = height * 2;

        double[][][] interpolatedImage = new double[depth][newHeight][width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < newHeight; i++) {
                for (int j = 0; j < width; j++) {
                    double interpolatedValue = 0.0;

                    for (int k = 0; k < weights[0].length; k++) {
                        int index = indices[i][k];
                        double weight = weights[i][k];
                        interpolatedValue += image[d][index-1][j] * weight;
                    }

                    interpolatedImage[d][i][j] = interpolatedValue;
                }
            }
        }

        return interpolatedImage;
    }
    public static double[][][] interpolateImageWidth(double[][][] image, double[][] weights, int[][] indices) {
        int depth = image.length;
        int height = image[0].length;
        int width = image[0][0].length;
        int newWidth = width * 2;

        double[][][] interpolatedImage = new double[depth][height][newWidth];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < newWidth; j++) {
                    double interpolatedValue = 0.0;

                    for (int k = 0; k < weights[0].length; k++) {
                        int index = indices[j][k];
                        double weight = weights[j][k];
                        interpolatedValue += image[d][i][index-1] * weight;
                    }

                    interpolatedImage[d][i][j] = interpolatedValue;
                }
            }
        }

        return interpolatedImage;
    }

    private static double cubic(double x) {
        double absX = Math.abs(x);
        double absX2 = absX * absX;
        double absX3 = absX2 * absX;

        double weight = 0;

        if (absX <= 1) {
            return 1.5 * absX3 - 2.5 * absX2 + 1;
        } else if (absX <= 2) {
            return -0.5 * absX3 + 2.5 * absX2 - 4 * absX + 2;
        } else {
            return 0;
        }

    }

    private static double[][][] calculateGradients(double[][][] imageStack) {
        int size = imageStack.length;
        int width = imageStack[0][0].length;
        int height = imageStack[0].length;
        double[][][] gradients = new double[2][height][width];
        for (int s = 0; s < size; s++) {
            double[][] image = imageStack[s];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double gx = getGradientX(image, x, y);
                    double gy = getGradientY(image, x, y);
                    gradients[0][y][x] += gx;
                    gradients[1][y][x] += gy;
                }
            }
        }
        return gradients;
    }

    private static double getGradientX(double[][] image, int x, int y) {
        int width = image[0].length;
        int left = x - 1;
        int right = x + 1;
        if (left < 0) {
            left = 0;
        }
        if (right >= width) {
            right = width - 1;
        }
        return image[y][right] - image[y][left];
    }

    private static double getGradientY(double[][] image, int x, int y) {
        int height = image.length;
        int top = y - 1;
        int bottom = y + 1;
        if (top < 0) {
            top = 0;
        }
        if (bottom >= height) {
            bottom = height - 1;
        }
        return image[bottom][x] - image[top][x];
    }

    private static double[][] averageGradients(double[][][] gradient) {
        int size = gradient.length;
        int height = gradient[0].length;
        int width = gradient[0][0].length;
        double[][] average = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0.0;
                for (int s = 0; s < size; s++) {
                    sum += gradient[s][y][x];
                }
                average[y][x] = sum / size;
            }
        }

        return average;
    }

    public static double[][] calculateVariance(double[][][] data, double[][] mean) {
        int size = data.length;
        int height = data[0].length;
        int width = data[0][0].length;
        double[][] variance = new double[height][width];

        for (int s = 0; s < size; s++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double diff = data[s][y][x] - mean[y][x];
                    variance[y][x] += diff * diff;
                }
            }
        }

        double scaleFactor = 1.0 / size;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                variance[y][x] *= scaleFactor;
            }
        }

        return variance;
    }


    private static double[][] calculateTotalGradientVariance(double[][] Gy_var, double[][] Gx_var) {
        int width = Gy_var[0].length;
        int height = Gy_var.length;
        double[][] variance = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                variance[y][x] = Gy_var[y][x] + Gx_var[y][x];
            }
        }
        return variance;
    }

    private static double[][] averageImageStack(double[][][] imageStack) {
        int size = imageStack.length;
        int width = imageStack[0][0].length;
        int height = imageStack[0].length;
        double[][] average = new double[height][width];
        for (int s = 0; s < size; s++) {
            double[][] image = imageStack[s];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    average[y][x] += image[y][x];
                }
            }
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                average[y][x] /= size;
            }
        }
        return average;
    }

    private static double[][] calculateImageVariance(double[][][] imageStack, double[][] average) {
        int size = imageStack.length;
        int width = imageStack[0][0].length;
        int height = imageStack[0].length;
        double[][] variance = new double[height][width];
        for (int s = 0; s < size; s++) {
            double[][] image = imageStack[s];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double diff = image[y][x] - average[y][x];
                    variance[y][x] += diff * diff;
                }
            }
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                variance[y][x] /= size;
            }
        }
        return variance;
    }

    private static double getMaxValue(double[][] image) {
        int width = image[0].length;
        int height = image.length;
        double max = image[0][0];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (image[y][x] > max) {
                    max = image[y][x];
                }
            }
        }
        return max;
    }

    private static int getMaxValue(int[][] image) {
        int width = image[0].length;
        int height = image.length;
        int max = image[0][0];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (image[y][x] > max) {
                    max = image[y][x];
                }
            }
        }
        return max;
    }

    private static double[][] normalizeImage(double[][] image, double maxValue) {
        int width = image[0].length;
        int height = image.length;
        double[][] normalized = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                normalized[y][x] = image[y][x] / maxValue;
            }
        }
        return normalized;
    }


    private static double[][] multiplyImages(double[][] image1) {
        int width = image1[0].length;
        int height = image1.length;
        double[][] result = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result[y][x] = image1[y][x] ;

            }
        }
        return result;
    }

    private static void saveModifiedImageStack(int[][][] imageStack, String savePath, String saveTitle) {
        int size = imageStack.length;
        int width = imageStack[0][0].length;
        int height = imageStack[0].length;
        String outputFile = savePath + saveTitle;
        try {
            ImageWriter writer = ImageIO.getImageWritersByFormatName("tif").next();
            ImageWriteParam writeParam = writer.getDefaultWriteParam();
            writeParam.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            writeParam.setCompressionQuality(1.0f);
            ImageOutputStream output = ImageIO.createImageOutputStream(new File(outputFile));
            writer.setOutput(output);
            writer.prepareWriteSequence(null);
            for (int s = 0; s < size; s++) {
                BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int gray = imageStack[s][y][x];
                        int rgb = (gray << 16) | (gray << 8) | gray;
                        image.setRGB(x, y, rgb);
                    }
                }
                IIOImage iioImage = new IIOImage(image, null, null);
                writer.writeToSequence(iioImage, writeParam);
            }
            writer.endWriteSequence();
            output.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static BufferedImage toBufferedImage(double[][] image) {
        int width = image[0].length;
        int height = image.length;
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (int) image[y][x];
                int rgb = (gray << 16) | (gray << 8) | gray;
                bufferedImage.setRGB(x, y, rgb);
            }
        }
        return bufferedImage;
    }


    // Convert image to 16-bit grayscale
    public static BufferedImage convertTo16BitGrayscale(double[][] image) {
        int height = image.length;
        int width = image[0].length;
        BufferedImage outputImage = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
        WritableRaster raster = outputImage.getRaster();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                raster.setSample(x, y, 0, (int) (image[y][x] * 65535));
            }
        }
        return outputImage;
    }

    private static void writeImageStack(double[][][] imageStack, String outputFolderPath, String filename) throws IOException {
        File outputFolder = new File(outputFolderPath);
        if (!outputFolder.exists()) {
            outputFolder.mkdirs();
        }

        int size = imageStack.length;
        for (int s = 0; s < size; s++) {
            String outputPath = outputFolderPath + "/" + s + "_" + filename;

            int width = imageStack[s][0].length;
            int height = imageStack[s].length;

            BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
            DataBufferUShort dataBuffer = (DataBufferUShort) bufferedImage.getRaster().getDataBuffer();
            short[] pixelData = dataBuffer.getData();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double pixelValue = imageStack[s][y][x];
                    short convertedValue = (short) Math.round(Math.max(0, Math.min(pixelValue, 65535)));
                    pixelData[y * width + x] = convertedValue;
                }
            }

            ImageIO.write(bufferedImage, "TIFF", new File(outputPath));
        }
    }
}

