from pyspark.sql import SparkSession
import os


os.environ['PYSPARK_PYTHON'] = r'C:\Users\AJ-PC\AppData\Local\Programs\Python\Python310\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\AJ-PC\AppData\Local\Programs\Python\Python310\python.exe'

def remove_nulls():

    spark = SparkSession.builder \
        .appName("RemoveNullValues") \
        .master("local[*]") \
        .getOrCreate()

    
    df_spark = spark.read.csv('your_file.csv', header=True, inferSchema=True)

 
    print("Original Data:")
    df_spark.show()


    df_cleaned = df_spark.dropna()


    print("Data After Removing Nulls:")
    df_cleaned.show()


    df_cleaned.write.csv('cleaned_data.csv', header=True, mode='overwrite')

  
    spark.stop()

if __name__ == "__main__":
    remove_nulls()
