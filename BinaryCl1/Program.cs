using Microsoft.ML;
using Microsoft.ML.AutoML;
using System;

namespace BinaryCl1
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            var mlContext = new MLContext();

            var trainingData = mlContext.Data.LoadFromTextFile<Models.SentimentIssue>("dataset/trainingdata.tsv", hasHeader: true);
            var testData = mlContext.Data.LoadFromTextFile<Models.SentimentIssue>("dataset/testData.tsv", hasHeader: true);

            ExperimentResult<Microsoft.ML.Data.BinaryClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateBinaryClassificationExperiment(10).Execute(trainingData);
            var model = experimentResult.BestRun.Model;

            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, scoreColumnName: "Score");
            mlContext.Model.Save(model, trainingData.Schema, "Model.zip");
        */
            var mlContext = new MLContext();
            DataViewSchema model;
           
            var transformer = mlContext.Model.Load("Model.zip",out model);
            PredictionEngine<Models.SentimentIssue, Models.SentimentPrediction> predictionEngine =
                mlContext.Model.CreatePredictionEngine<Models.SentimentIssue, Models.SentimentPrediction>(transformer);

           var prediction  = predictionEngine.Predict(new Models.SentimentIssue() { 
                Text="I hate this worst application "
            });
            Console.WriteLine("My Prediction:" + prediction.Prediction);
        }
    }
}
