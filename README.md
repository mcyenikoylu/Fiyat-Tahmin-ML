# Fiyat-Tahmin-ML
Microsoft machine learning ile fiyat tahmini. {.NET CORE for Mac OS} [ML.NET Tutorial]

dotnet new console -o consumeModelApp

export PATH=$HOME/.dotnet/tools:$PATH

## Model
vendor_id,
rate_code,
passenger count,
trip time in secs,
trip distance,
payment_type,
fare_amount

mlnet auto-train --task multiclass-classification --dataset "taxi-fare-train.csv" --label-column-name "rate_code" --max-exploration-time 10