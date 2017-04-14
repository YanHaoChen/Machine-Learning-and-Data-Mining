import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class pm25_v4 {
    
    public static class Map extends Mapper<LongWritable, Text,Text, MapWritable> {
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
    public static class Reduce extends Reducer<Text, MapWritable, Text, Text> {
        
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
            for (int i=0;i<24;i++){
                float avg = (float)map.get(i) / (float)count;
                
                output += String.valueOf(i) + "/" + String.format("%.2f",avg)+" ";
            }
            
            context.write(key , new Text(output));
        }
    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        
        Job job = new Job(conf, "pm25_v4");
        
        
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);
        job.setJarByClass(pm25_v4.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        job.setMapperClass(Map.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.waitForCompletion(true);
    }
    
}
