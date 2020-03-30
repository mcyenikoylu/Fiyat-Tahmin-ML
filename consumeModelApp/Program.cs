using System;
using SampleMulticlassClassification.Model.DataModels;
using Microsoft.ML;

namespace consumeModelApp
{
    class Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("Hello World!");
            ConsumeModel();
        }
        public static void ConsumeModel()
        {
            // Load the model
            MLContext mlContext = new MLContext();

            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use the code below to add input data
            var input = new ModelInput();
            //input.SentimentText = "That is rude";

// vendor_id,
// rate_code,
// passenger count,
// trip time in secs,
// trip distance,
// payment_type,
// fare_amount
            //CMT,3,1,2528,18.1,CSH,71
            
            input.Rate_code = 3;
    //        input.Passenger_count = 1;
//            input.Trip_time_in_secs = 2528;
//input.Trip_distance = float.Parse("18.1");
input.Payment_type = "CSH";
//input.Fare_amount = 71;

            // Try model on sample data
            // True is toxic, false is non-toxic
            ModelOutput result = predEngine.Predict(input);

            //Console.WriteLine($"satıcı kimliği: {input.Vendor_id} | Oran Kodu: {input.Rate_code} | Yolcu Sayisi: {input.Passenger_count} | saniye olarak yolculuk süresi: {input.Trip_time_in_secs} | yolculuk mesafesi: {input.Trip_distance} | ödeme şekli: {input.Payment_type} | Hesap toplamı: {input.Fare_amount}");
        Console.WriteLine($"Tahmin: {(Convert.ToBoolean(result.Prediction) ? "Olumlu" : "Olumsuz")} yolcu");
        }
    }
}
