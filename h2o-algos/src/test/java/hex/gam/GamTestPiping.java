package hex.gam;

import hex.DataInfo;
import hex.gam.GAMModel.GAMParameters.BSType;
import hex.glm.GLMModel;
import hex.glm.GLMTask;
import org.junit.BeforeClass;
import org.junit.Test;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;

/***
 * Here I am going to test the following:
 * - model matrix formation with centering
 */
public class GamTestPiping extends TestUtil {
  @BeforeClass
  public static void setup() {
    stall_till_cloudsize(1);
  }

  /**
   * This test is to make sure that we carried out the expansion of ONE gam column to basis functions
   * correctly with and without centering.  I will compare the following with R runs:
   * 1. binvD generation;
   * 2. model matrix that contains the basis function value for each role of the gam column
   * 3. penalty matrix without centering
   * 4. model matrix after centering
   * 5. penalty matrix with centering.
   * 
   * I compared my results with the ones generated from R mgcv library.  At this point, I am not concerned with
   * model content, only the data transformation.
   * 
   */
  @Test
  public void test1GamTransform() {
    try {
      Scope.enter();
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[] gamCols = new String[]{"C6"};
      double[][] knots = new double[1][];
      knots[0] = new double[]{-1.99905699, -0.98143075, 0.02599159, 1.00770987, 1.99942290};
      GAMModel model = getModel(GLMModel.GLMParameters.Family.gaussian,
              Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv")),
              "C11", gamCols, ignoredCols, new int[]{5}, new BSType[]{BSType.cr}, false, 
              true, knots, new double[]{1}, new double[]{0}, new double[]{0});
      Scope.track_generic(model);
      double[][] rBinvD = new double[][]{{1.5605080,
              -3.5620961, 2.5465468, -0.6524143, 0.1074557}, {-0.4210098, 2.5559955, -4.3258597, 2.6228736,
              -0.4319995}, {0.1047194, -0.6357626, 2.6244918, -3.7337994, 1.6403508}};
      double[][] rS = new double[][]{{0.078482471, -0.17914814,  0.1280732, -0.03281181,  0.005404258}, {-0.179148137,
              0.48996127, -0.4772073,  0.19920397, -0.032809824},
              {0.128073216, -0.47720728,  0.7114729, -0.49778106,  0.135442250},
              {-0.032811808,  0.19920397, -0.4977811,  0.52407922, -0.192690331},
              { 0.005404258, -0.03280982,  0.1354423, -0.19269033,  0.084653646}};
      double[][] rScenter = new double[][]{{0.5448733349, -0.4278253,  0.1799158, -0.0009387815}, {-0.4278253091,
              0.7554630, -0.5087565,  0.1629793339}, {0.1799157665, -0.5087565,  0.4339205, -0.1867776214}, 
              {-0.0009387815,  0.1629793, -0.1867776,  0.1001323668}};

      TestUtil.checkDoubleArrays(model._output._binvD[0], rBinvD, 1e-1); // compare binvD generation
      TestUtil.checkDoubleArrays(model._output._penaltyMatrices[0], rS, 1e-1);  // compare penalty terms
      TestUtil.checkDoubleArrays(model._output._penaltyMatrices_center[0], rScenter, 1e-1);

      Frame transformedData = ((Frame) DKV.getGet(model._output._gamTransformedTrain));  // compare model matrix
      Scope.track(transformedData);
      Scope.track(transformedData.remove("C11"));
      Frame rTransformedData = parse_test_file("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C6Gam.csv");
      Scope.track(rTransformedData);
      TestUtil.assertIdenticalUpToRelTolerance(transformedData, rTransformedData, 1e-1);

      Frame transformedDataC = ((Frame) DKV.getGet(model._output._gamTransformedTrainCenter));  // compare model matrix with centering
      Scope.track(transformedDataC);
      Scope.track(transformedDataC.remove("C11"));
      for (String cname : ignoredCols)
        Scope.track(transformedDataC.remove(cname));
      Frame rTransformedDataC = parse_test_file("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C6Gam_center.csv");
      Scope.track(rTransformedDataC);
      TestUtil.assertIdenticalUpToRelTolerance(transformedDataC, rTransformedDataC, 1e-1);
    } finally {
      Scope.exit();
    }
  }


  /**
   * This is the same test as in test1GamTransform except in this case we check out three gam columns instead of one.
   * In addition, we compare 
   * 1. BinvD;
   * 2. penalty matrices after centering
   * 3. model matrix after centering.
   */
  @Test
  public void test3GamTransform() {
    try {
      Scope.enter();
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[] gamCols = new String[]{"C6", "C7", "C8"};
      int numGamCols = gamCols.length;
      double[][] knots = new double[][]{{-1.99905699, -0.98143075, 0.02599159, 1.00770987, 1.99942290},{-1.999821861,
              -1.005257990, -0.006716042, 1.002197392, 1.999073589}, {-1.999675688, -0.979893796, 0.007573327,
              1.011437347, 1.999611676}};
      GAMModel model = getModel(GLMModel.GLMParameters.Family.gaussian,
              Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"))
              , "C11", gamCols, ignoredCols, new int[]{5,5,5}, new BSType[]{BSType.cr,BSType.cr,BSType.cr},
              false, true, knots, new double[]{1,1,1}, new double[]{0,0,0}, new double[]{0,0,0});
      Scope.track_generic(model);
      double[][][] rBinvD = new double[][][]{{{1.5605080,
              -3.5620961, 2.5465468, -0.6524143, 0.1074557}, {-0.4210098, 2.5559955, -4.3258597, 2.6228736,
              -0.4319995}, {0.1047194, -0.6357626, 2.6244918, -3.7337994, 1.6403508}}, {{1.6212347, -3.6647127,
              2.5744837, -0.6390045, 0.1079989},{-0.4304170,  2.5705123, -4.2598869,  2.5509266, -0.4311351},{0.1082500,
              -0.6464846,  2.5538194,-3.6243725, 1.6087877}},{{1.5676838, -3.6153068,  2.5754978, -0.6358007, 
              0.1079259 },{-0.4150543,  2.5862931, -4.3172873 , 2.5848160, -0.4387675},{0.1045808, -0.6516658,  
              2.5880211,-3.6755096,  1.6345735}}};
      double[][][] rScenter = new double[][][]{{{0.5448733349, -0.4278253,  0.1799158, -0.0009387815}, {-0.4278253091,
              0.7554630, -0.5087565,  0.1629793339}, {0.1799157665, -0.5087565,  0.4339205, -0.1867776214},
              {-0.0009387815,  0.1629793, -0.1867776,  0.1001323668}},{{0.5697707675, -0.4362176,  0.1755907,
              -0.0001031409},{-0.4362176085, 0.7592479, -0.4915641, 0.1637712460},{0.1755907088, -0.4915641,  0.4050463,
              -0.1811556223},{-0.0001031409, 0.1637712, -0.1811556, 0.1003644978}},{{0.550472199, -0.4347752, 0.1714131,
              -0.001386494},{-0.434775190, 0.7641889, -0.4932521, 0.165206427},{0.171413072, -0.4932521, 0.4094752,
              -0.184101847},{-0.001386494, 0.1652064, -0.1841018, 0.101590357}}};

      for (int test=0; test < numGamCols; test++) {
        TestUtil.checkDoubleArrays(model._output._binvD[test], rBinvD[test], 1e-6); // compare binvD generation
        TestUtil.checkDoubleArrays(model._output._penaltyMatrices_center[test], rScenter[test], 1e-6);
      }

      Frame transformedDataC = ((Frame) DKV.getGet(model._output._gamTransformedTrainCenter));  // compare model matrix with centering
      Scope.track(transformedDataC);
      Scope.track(transformedDataC.remove("C11"));
      for (String cname : ignoredCols)
        Scope.track(transformedDataC.remove(cname));
      Frame rTransformedDataC = parse_test_file("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C6Gam_center.csv");
      rTransformedDataC.add(Scope.track(parse_test_file("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C7Gam_center.csv")));
      rTransformedDataC.add(Scope.track(parse_test_file("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C8Gam_center.csv")));
      Scope.track(rTransformedDataC);
      TestUtil.assertIdenticalUpToRelTolerance(transformedDataC, rTransformedDataC, 1e-2);
    } finally {
      Scope.exit();
    }
  }

  
  public GAMModel getModel(GLMModel.GLMParameters.Family family, Frame train, String responseColumn,
                           String[] gamCols, String[] ignoredCols, int[] numKnots, BSType[] bstypes, boolean saveZmat,
                           boolean savePenalty, double[][] knots, double[] scale, double[] alpha, double[] lambda) {
    GAMModel gam = null;
    try {
      Scope.enter();
      // set cat columns
      int numCols = train.numCols();
      int enumCols = (numCols - 1) / 2;
      for (int cindex = 0; cindex < enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      int response_index = numCols - 1;
      if (family.equals(GLMModel.GLMParameters.Family.binomial) || (family.equals(GLMModel.GLMParameters.Family.multinomial))) {
        train.replace((response_index), train.vec(response_index).toCategoricalVec()).remove();
      }
      DKV.put(train);
      Scope.track(train);

      GAMModel.GAMParameters params = new GAMModel.GAMParameters();
      params._standardize = false;
      params._scale = scale;
      params._family = family;
      params._response_column = responseColumn;
      params._train = train._key;
      params._bs = bstypes;
      params._k = numKnots;
      params._ignored_columns = ignoredCols;
      params._alpha = alpha;
      params._lambda = lambda;
      params._compute_p_values = true;
      params._gam_X = gamCols;
      params._train = train._key;
      params._family = family;
      params._knots = knots;
      params._link = GLMModel.GLMParameters.Link.family_default;
      params._saveZMatrix = saveZmat;
      params._saveGamCols = true;
      params._savePenaltyMat = savePenalty;
      gam = new GAM(params).trainModel().get();
    } finally {
      Scope.exit();
    }
    return gam;
  }

  /**
   * The following test makes sure that we have added the smooth term to the gradient, hessian and likelihood/objective
   * calculation to GAM
   */
  @Test
  public void testGradientTask() {
    try {
      Scope.enter();
      String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[] gamCols = new String[]{"C6", "C7", "C8"};
      int numGamCols = gamCols.length;
      double[][] knots = new double[][]{{-1.99905699, -0.98143075, 0.02599159, 1.00770987, 1.99942290},{-1.999821861,
              -1.005257990, -0.006716042, 1.002197392, 1.999073589}, {-1.999675688, -0.979893796, 0.007573327,
              1.011437347, 1.999611676}};
      GAMModel model = getModel(GLMModel.GLMParameters.Family.gaussian,
              Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv")), "C11",
              gamCols, ignoredCols, new int[]{5,5,5}, new BSType[]{BSType.cr,BSType.cr,BSType.cr}, false, 
              true, knots, new double[]{1,1,1}, new double[]{0,0,0}, new double[]{0,0,0});
      Scope.track_generic(model);      
      double[] beta = model._output._model_beta;  // start model here
      Frame dataFrame = ((Frame) DKV.getGet(model._output._gamTransformedTrainCenter)); // actual data used
      for (String cname : ignoredCols)
        Scope.track(dataFrame.remove(cname));
      Scope.track(dataFrame);
      GLMModel.GLMParameters glmParms = setGLMParamsFromGamParams(model._parms, dataFrame);
      double[][][] penalty_mat = model._output._penaltyMatrices_center;
      DataInfo dinfo = new DataInfo(dataFrame, null, 1,
              glmParms._use_all_factor_levels || glmParms._lambda_search, glmParms._standardize ? 
              DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE, DataInfo.TransformType.NONE, 
              true, false, false, false, false, false);
      DKV.put(dinfo._key, dinfo);
      Scope.track_generic(dinfo);
      GLMTask.GLMGaussianGradientTask gradientTaskNoPenalty = (GLMTask.GLMGaussianGradientTask) 
              new GLMTask.GLMGaussianGradientTask(null,dinfo, glmParms,1,
                      beta).doAll(dinfo._adaptedFrame);
      int[][] gamCoeffIndices = new int[][]{{},{},{}};
      GLMTask.GLMGaussianGradientTask gradientTaskPenalty = (GLMTask.GLMGaussianGradientTask)
              new GLMTask.GLMGaussianGradientTask(null,dinfo, glmParms,1,
                      beta, model._output._penaltyMatrices_center, gamCoeffIndices).doAll(dinfo._adaptedFrame);




    } finally {
      Scope.exit();
    }
  }
  
  public GLMModel.GLMParameters setGLMParamsFromGamParams(GAMModel.GAMParameters gamP, Frame trainFrame) {
    GLMModel.GLMParameters parms = new GLMModel.GLMParameters();
    parms._family = gamP._family;
    parms._response_column = gamP._response_column;
    parms._train = trainFrame._key;
    parms._ignored_columns = gamP._ignored_columns;
    return parms;
  }
  
  @Test
  public void testStandardizedCoeff() {
    String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15",
            "C17", "C18", "C19", "C20"};
    String[] gamCols = new String[]{"C11", "C12"};
    // test for Gaussian
    testCoeffs(GLMModel.GLMParameters.Family.gaussian, "smalldata/glm_test/gaussian_20cols_10000Rows.csv",
            "C21", gamCols, ignoredCols);
    // test for binomial
    testCoeffs(GLMModel.GLMParameters.Family.binomial, "smalldata/glm_test/binomial_20_cols_10KRows.csv",
            "C21", gamCols, ignoredCols);
    // test for multinomial
    ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
    gamCols = new String[]{"C6", "C7"};
    testCoeffs(GLMModel.GLMParameters.Family.multinomial,
            "smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv", "C11",
            gamCols, ignoredCols);

  }

  public void testCoeffs(GLMModel.GLMParameters.Family family, String fileName, String responseColumn,
                         String[] gamCols, String[] ignoredCols) {
    try {
      Scope.enter();
      Frame train = parse_test_file(fileName);
      // set cat columns
      int numCols = train.numCols();
      int enumCols = (numCols - 1) / 2;
      for (int cindex = 0; cindex < enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      int response_index = numCols - 1;
      if (family.equals(GLMModel.GLMParameters.Family.binomial) || (family.equals(GLMModel.GLMParameters.Family.multinomial))) {
        train.replace((response_index), train.vec(response_index).toCategoricalVec()).remove();
      }
      DKV.put(train);
      Scope.track(train);

      GAMModel.GAMParameters params = new GAMModel.GAMParameters();
      params._standardize = false;
      params._family = family;
      params._response_column = responseColumn;
      params._train = train._key;
      params._bs = new BSType[]{BSType.cr, BSType.cr};
      params._k = new int[]{6, 6};
      params._ignored_columns = ignoredCols;
      params._gam_X = gamCols;
      params._train = train._key;
      params._family = family;
      params._link = GLMModel.GLMParameters.Link.family_default;
      params._saveZMatrix = true;
      params._saveGamCols = true;
      params._standardize = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      Frame transformedData = ((Frame) DKV.getGet(gam._output._gamTransformedTrain));
      Scope.track(transformedData);
      numCols = transformedData.numCols() - 1;
      for (int ind = 0; ind < numCols; ind++)
        System.out.println(transformedData.vec(ind).mean());
      Frame transformedDataCenter = ((Frame) DKV.getGet(gam._output._gamTransformedTrainCenter));
      Scope.track(transformedDataCenter);
      numCols = transformedDataCenter.numCols() - 1;
      System.out.println("Print center gamx");
      for (int ind = 0; ind < numCols; ind++)
        System.out.println(transformedDataCenter.vec(ind).mean());
      Frame predictF = gam.score(transformedData); // predict with train data
      Scope.track(predictF);
      Frame predictRaw = gam.score(train); // predict with train data
      Scope.track(predictRaw);
      TestUtil.assertIdenticalUpToRelTolerance(predictF, predictRaw, 1e-6);
    } finally {
      Scope.exit();
    }
  }


}
