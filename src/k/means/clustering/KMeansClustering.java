package k.means.clustering;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.io.StringReader;
import java.util.Arrays;
import java.util.Scanner;

public class KMeansClustering {
    // Data members

    private double[][] _data; // Array of all records in dataset
    private int[] _label;  // generated cluster labels

    // by comparing _label and _withLabel, we can compute accuracy. 
    // However, the accuracy function is not defined yet.
    private double[][] _centroids; // centroids: the center of clusters
    private int _nrows, _ndims; // the number of rows and dimensions
    private int _numClusters; // the number of clusters;

    // Constructor; loads records from file <fileName>. 
    public KMeansClustering(String fileName) {
        String line = null;
        String[] array;
        // Creates a new KMeans object by reading in all of the records that are stored in a txt file
        try {
            // FileReader reads text files in the default encoding.
            FileReader fileReader
                    = new FileReader(new File(System.getProperty("user.dir") + File.separator + fileName));

            // Always wrap FileReader in BufferedReader.
            BufferedReader bufferedReader
                    = new BufferedReader(fileReader);
            _ndims = bufferedReader.readLine().split("\t").length - 1;
            while ((line = bufferedReader.readLine()) != null) {
                _nrows++;
            }

            // initialize the _data variable
            _data = new double[_nrows][];
            for (int i = 0; i < _nrows; i++) {
                _data[i] = new double[_ndims];
            }

            // read records from the txt file
            int nrow = 0;
            fileReader
                    = new FileReader(new File(System.getProperty("user.dir") + File.separator + fileName));
            bufferedReader
                    = new BufferedReader(fileReader);
            while ((line = bufferedReader.readLine()) != null && nrow < _nrows) {
                array = line.split("\t");
                double[] dv = new double[_ndims];
                for (int i = 0; i < _ndims; i++) {
                    dv[i] = Double.parseDouble(array[i]);
                }
                _data[nrow] = dv;
                nrow++;
            }

            System.out.println("loaded data");

            // Always close files.
            bufferedReader.close();
        } catch (FileNotFoundException ex) {
            System.out.println(
                    "Unable to open file '"
                    + fileName + "'");
        } catch (IOException ex) {
            System.out.println(
                    "Error reading file '"
                    + fileName + "'");
        }

    }

    // Perform k-means clustering with the specified number of clusters and
    // Eucliden distance metric. 
    // centroids are the initial centroids. It is optional. If set to null, the initial centroids will be generated randomly.
    public void clustering(int numClusters, double[][] centroids) {
        _numClusters = numClusters;
        if (centroids != null) {
            _centroids = centroids;
        } else {
            // randomly selected centroids
            _centroids = new double[_numClusters][];

            ArrayList idx = new ArrayList();
            for (int i = 0; i < numClusters; i++) {
                int c;
                do {
                    c = (int) (Math.random() * _nrows);
                } while (idx.contains(c)); // avoid duplicates
                idx.add(c);
                // copy the value from _data[c]
                _centroids[i] = new double[_ndims];
                for (int j = 0; j < _ndims; j++) {
                    _centroids[i][j] = _data[c][j];
                }
            }
            System.out.println("selected random centroids");

        }

        double[][] c1 = new double[_numClusters][_ndims];
        double threshold = 0.001;
        int round = 0;

        do {
            //assign record to the closest centroid
            _label = new int[_nrows];
            for (int i = 0; i < _nrows; i++) {
                _label[i] = closest(_data[i]);
            }

            System.out.println("\n\nAt this step");
            System.out.println("Value of clusters");
            for (int ci = 0; ci < _numClusters; ++ci) {
                System.out.print("K" + (ci + 1) + " : [ ");
                for (int i = 0; i < _nrows; ++i) {
                    if (ci == _label[i]) {
                        System.out.print("{");
                        for (int j = 0; j < _ndims; ++j) {
                            System.out.print(_data[i][j] + " ");
                        }
                        System.out.print("} ");
                    }
                }
                System.out.println("]");
            }

            System.out.println("Value of centroids ");
            for (int i = 0; i < _numClusters; ++i) {
                System.out.print("C" + (i + 1) + " = {");
                for (int j = 0; j < _ndims; ++j) {
                    System.out.print(_centroids[i][j] + "  ");
                }
                System.out.println("}");
            }

            // backup old centroid
            c1 = _centroids;
            // recompute centroids based on the assignments  
            _centroids = updateCentroids();
            round++;
        } while (!converge(_centroids, c1));

        System.out.println("\n\n\nThe Final Clusters By Kmeans are as follows: ");
        for (int ci = 0; ci < _numClusters; ++ci) {
            System.out.print("K" + (ci + 1) + " : [ ");
            for (int i = 0; i < _nrows; ++i) {
                if (ci == _label[i]) {
                    System.out.print("{");
                    for (int j = 0; j < _ndims; ++j) {
                        System.out.print(_data[i][j] + " ");
                    }
                    System.out.print("} ");
                }
            }
            System.out.println("]");
        }

        System.out.println("\n\nClustering converges at round " + round);
    } 

    // find the closest centroid for the record v 
    private int closest(double[] v) {
        double mindist = dist(v, _centroids[0]);
        int label = 0;
        for (int i = 1; i < _numClusters; i++) {
            double t = dist(v, _centroids[i]);
            if (mindist > t) {
                mindist = t;
                label = i;
            }
        }
        return label;
    }

    // compute Euclidean distance between two vectors v1 and v2
    private double dist(double[] v1, double[] v2) {
        double sum = 0;
        for (int i = 0; i < _ndims; i++) {
            double d = v1[i] - v2[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    // according to the cluster labels, recompute the centroids 
    // the centroid is updated by averaging its members in the cluster.
    // this only applies to Euclidean distance as the similarity measure.
    private double[][] updateCentroids() {
        // initialize centroids and set to 0
        double[][] newc = new double[_numClusters][]; //new centroids 
        int[] counts = new int[_numClusters]; // sizes of the clusters

        // intialize
        for (int i = 0; i < _numClusters; i++) {
            counts[i] = 0;
            newc[i] = new double[_ndims];
            for (int j = 0; j < _ndims; j++) {
                newc[i][j] = 0;
            }
        }

        for (int i = 0; i < _nrows; i++) {
            int cn = _label[i]; // the cluster membership id for record i
            for (int j = 0; j < _ndims; j++) {
                newc[cn][j] += _data[i][j]; // update that centroid by adding the member data record
            }
            counts[cn]++;
        }

        // finally get the average
        for (int i = 0; i < _numClusters; i++) {
            for (int j = 0; j < _ndims; j++) {
                newc[i][j] /= counts[i];
            }
        }

        return newc;
    }

    // check convergence condition
    // max{dist(c1[i], c2[i]), i=1..numClusters
    private boolean converge(double[][] c1, double[][] c2) {
        // c1 and c2 are two sets of centroids 
        double maxv = 0;
        for (int i = 0; i < _numClusters; i++) {
            for (int j = 0; j < _ndims; j++) {
                if (c1[i][j] != c2[i][j]) {
                    return false;
                }
            }
        }
        return true;

    }

    public double[][] getCentroids() {
        return _centroids;
    }

    public int[] getLabel() {
        return _label;
    }

    public int nrows() {
        return _nrows;
    }

    public static void main(String[] astrArgs) {
        /**
         * The code commented out here is just an example of how to use the
         * provided functions and constructors.
         *
         */
        //c1= {43.91304347826087  146.04347826086956  }
        //c2= {69.28571428571429  18.642857142857142  }
        //c3= {20.15  64.95  }
        //c4= {98.17647058823529  114.88235294117646  }
        KMeansClustering KM = new KMeansClustering("ruspini.txt");
        Scanner scr = new Scanner(System.in);
        /* Accepting num of clusters */
        System.out.print("Enter the number of clusters: ");
        int p = scr.nextInt();
        System.out.print("Do you want to input centroid (y/n)? ");
        if (scr.next().equals("y")) {
            double[][] c = new double[p][2];
            for (int i = 0; i < p; i++) {
                System.out.print("Input Centroid"+(i+1)+" (ex. 2,3): ");
                c[i]=Arrays.stream(scr.next().split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
            }
            KM.clustering(p, c);
        }else{
            KM.clustering(p, null);
        }
    }
}
