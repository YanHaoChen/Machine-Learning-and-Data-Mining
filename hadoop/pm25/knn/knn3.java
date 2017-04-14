import java.io.IOException;
import java.util.*;
import java.lang.Math;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class knn3 {
    public static class Map1 extends Mapper<LongWritable, Text,Text, MapWritable> {
        HashMap<String, int[]> dataMap = new HashMap<String, int[]>();
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            
            String line = value.toString();
            String[] valueArray = line.split(",");
            String station = valueArray[1];
            
            if (valueArray[2].equals("PM2.5")){
                if(dataMap.get(station)==null){
                    int[] temp = new int[25];
                    dataMap.put(station, temp);
                }
                int[] pm = dataMap.get(station);
                for(int i=3;i<27;i++){
                    try{
                        pm[i-3] += Integer.parseInt(valueArray[i]);
                    } catch (Exception ex) {
                        return;
                    }
                    
                }
                pm[24] +=1;
                dataMap.put(station, pm);
            }
        }
        public void cleanup(Context context) throws IOException, InterruptedException {
            for(String dataKey : dataMap.keySet()){
                MapWritable arry = new MapWritable();
                int[] temp = dataMap.get(dataKey);
                for(int i=0;i<25;i++){
                    arry.put(new IntWritable(i),new IntWritable(temp[i]));
                }
                context.write(new Text(dataKey), arry);
            }
        }
    }
    public static class Reduce1 extends Reducer<Text, MapWritable, Text, Text> {
        public void reduce(Text key, Iterable<MapWritable>arry, Context context)
        throws IOException, InterruptedException {
            HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
            int count = 0;
            
            for (int i=0;i<24;i++){
                map.put(i,0);   
            }       
            for (MapWritable ar:arry) {
                for (int i=0; i<24;i++){
                    int pm25 = ((IntWritable)ar.get(new IntWritable(i))).get();
                    map.put(i, map.get(i)+ pm25);
                }
                count += ((IntWritable)ar.get(new IntWritable(24))).get();
            }
            
            String output = "";
            for (int i=0;i<23;i++){
                float avg = (float)map.get(i) / (float)count;
                output +=String.format("%.2f",avg)+",";
            }
            float avg = (float)map.get(23) / (float)count;
            output +=String.format("%.2f",avg);
            context.write(key , new Text(output));
        }
    }

    public static class Map2 extends Mapper<LongWritable, Text,Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
              
            context.write(new Text("1"), value);
        }
    }
    public static class Reduce2 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text>datas, Context context)
        throws IOException, InterruptedException {
            
            ArrayList<String> all_datas = new ArrayList<String>();
            for (Text data:datas) {
                String station_and_values = data.toString();
                all_datas.add(station_and_values);
            }
            Collections.sort(all_datas);

            String output_string = "";
            for (int i =0; i < all_datas.size()-1;i++){
                output_string += all_datas.get(i) + "_";
            }
            output_string += all_datas.get(all_datas.size()-1);
            
            for (String station:all_datas) {
                context.write(new Text(station+"&"), new Text(output_string));    
            }
        }
    }
    public static class Map3 extends Mapper<LongWritable, Text,Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            
            String line = value.toString();
            String[] stations_and_values = line.split("&");
            String station = stations_and_values[0];
            String station_name = station.split("\t")[0];
            String[] station_values = station.split("\t")[1].split(",");
            float[] station_values_float = new float[24];

            for (int i = 0; i<24 ;i++ ){
                station_values_float[i] = Float.parseFloat(station_values[i]);
            }
            
            String other_station_and_values_string = stations_and_values[1];
            String other_station_and_values_string_del_tab =  other_station_and_values_string.substring(1,other_station_and_values_string.length());
            String[] other_stations_and_values = other_station_and_values_string_del_tab.split("_");

            String[] results_station = new String[76];
            float[] results = new float[76];

            
            int j = 0;
            for (String other_station_and_value : other_stations_and_values){
                String[] other_values = other_station_and_value.split("\t")[1].split(",");
                float this_station_sq = (float)0;
                for(int i = 0;i<24; i++){
                    float other_value = Float.parseFloat(other_values[i]);
                    this_station_sq += (float)Math.pow(station_values_float[i]-other_value, 2); 
                }
                results[j] = (float)Math.pow(this_station_sq, 0.5);
                results_station[j] = other_station_and_value.split("\t")[0];
                j++;
            }
            int outputn = 3;
            for (int i=0;i < outputn+1; i++){
                for(int y=i+1;y< results.length;y++){
                    if (results[i] > results[y]){
                        float tempf = results[i];
                        String temps = results_station[i];
                        results[i] = results[y];
                        results_station[i] = results_station[y];
                        results[y] = tempf;
                        results_station[y] = temps;
                    }
                }
            }
            String output = "";
            for (int i=1;i< outputn+1;i++){
                output+= String.format("%s,%.2f ",results_station[i],results[i]);
            }
            context.write(new Text(station_name), new Text(output));
        }
    }
    public static class Reduce3 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text>results, Context context)
        throws IOException, InterruptedException {
            for(Text result:results){
                context.write(key, new Text(result));
           }
        }
    }
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        
        Job job = new Job(conf, "knn3");
        
        job.setMapperClass(Map1.class);
        job.setReducerClass(Reduce1.class);
        job.setJarByClass(knn3.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.waitForCompletion(true);
        
        Job job2 = new Job(conf,"knn3");  
        
        job2.setMapperClass(Map2.class);
        job2.setReducerClass(Reduce2.class);
        job2.setJarByClass(knn3.class);
        
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        
        job2.setInputFormatClass(TextInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job2,new Path(args[1]));  
        FileOutputFormat.setOutputPath(job2,new Path(args[2]));  
        job2.waitForCompletion(true);

        Job job3 = new Job(conf,"knn3");  
        
        job3.setMapperClass(Map3.class);
        //job3.setReducerClass(Reduce3.class);
        job3.setJarByClass(knn3.class);
        
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        
        job3.setInputFormatClass(TextInputFormat.class);
        job3.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job3,new Path(args[2]));  
        FileOutputFormat.setOutputPath(job3,new Path(args[3]));  
        job3.waitForCompletion(true);  
    }
}