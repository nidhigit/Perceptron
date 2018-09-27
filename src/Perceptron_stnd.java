

import java.util.*; 

import java.io.*;
import java.util.*;
import java.nio.*;
public class Perceptron_stnd {
	
	
 	public static void main(String args[])
	{
 		Data data[]=new Data[1000];
 		Double w[] = new Double[20];  
 		
 		int index=-1;
		File file = new File("C:\\Machine Learning\\hw2\\dataset\\diabetes.train");
		 try {
		      FileInputStream fis = new FileInputStream(file);
		      BufferedReader df    = new BufferedReader(new InputStreamReader(fis));
		      String line;
	//	      HashMap <String,List> attribute_ValueSet = new HashMap();
 		
	//	      String labels_in_sample[]=new String[7000];
		      int y = 0 ;
		      HashMap<Integer, Double> x;
		   
		      
		      // read the training set and create a Dataset
		      
		      	while ((line = df.readLine())!=null){
 		      			String[] result = line.split("\\ ");
 		 		       x = new HashMap<Integer, Double>();

		      				for (int i=0; i<result.length; i++) {
		      						
		      						if(i==0) //it's a label, create extra 'x' for bias.
		      						{
		      							 y =  Integer.valueOf(result[i]);
		      							 x.put(0,(double)1);
		      						}
		      						else
		      						{
		      							String[] result1 = result[i].split("\\:");
		      							Integer key = Integer.valueOf(result1[0]);
		      							Double value = Double.valueOf(result1[1]);
		      							x.put(key,value);	
		      						}
		      				}
		      			data[++index] = new Data(x,y);	
	 		    }
		      	df.close();

 		   }	catch (IOException err) {
 			   		err.printStackTrace();
		    	}
		 
		 	runSimplePerceptron(data,index,w);
		 	runDecayingRatePerceptron(data,index,w);
	  
	}
 	
 	public static Double[] learnSimplePerceptron(Data data1[],int index,double LearningRate)
 	{
 		Double w[] = new Double[20];  
 		Double min =-0.01;  //  Set To Your Desired Min Value
	    Double max = 0.01;
 		double smallRandNumber = min + new Random().nextDouble() * (max - min); 
 		Arrays.fill(w, smallRandNumber);
 		
 		Data data[] = Arrays.copyOf(data1,index);
 		
 		//define rate of change and pick one randomly
 //		double LearningRates[] = {1, 0.1, 0.01};
 //		int minInt =0;  //  Set To Your Desired Min Value
//	    int maxInt =2;
// 		Random rand = new Random();
 		
// 	 	int randomNum = rand.nextInt((maxInt - minInt) + 1) + minInt;
 	
 	// 	double LearningRate = LearningRates[randomNum];
 	//	double LearningRate = (double)0.1;
 		//check the sign, if -ve then update vector 'w' and bias 'b' as needed.
 	//	int k=0;
 	//	int seed=2500;
 	//	while(k++<seed)
 		
 		
 		//shuffle the data..
 	 	ArrayList<Data> arrayList = new ArrayList<Data>(Arrays.asList(Arrays.copyOf(data,index))); 
 		Collections.shuffle(arrayList);
 		data = arrayList.toArray(new Data[arrayList.size()]);
 	  		
 		for(int i=0;i<index;i++)
 		{
 			
 	//		int minInt1 =0;  //  Set To Your Desired Min Value
 	//	    int maxInt1 =index-1;
 		//    int i=0;
 	 	//	Random rand1= new Random();
 	 		
 	 //	 	i = rand1.nextInt((maxInt1 - minInt1) + 1) + minInt;
 			
 			double trueLabel;
 			double predictedLabel;
 			double dotProduct_wT_x = 0.0f;
 			HashMap<Integer, Double> x = new HashMap<Integer, Double>();
 		 
 			trueLabel = data[i].getY();
 			x= data[i].getX();
 			
 			//wT*x (contains bias)
 			for (Map.Entry<Integer, Double> entry : x.entrySet()) {
 				dotProduct_wT_x = dotProduct_wT_x +  w[entry.getKey()] * entry.getValue();
 			}
 			
 			//predict the label
 			if(dotProduct_wT_x < 0)
 				predictedLabel = -1;
 			else	
 				predictedLabel = +1;
 			
 			//check if prediction is correct, if not update the weight vector and bias.
 			if(predictedLabel != trueLabel)
 		//	if(dotProduct_wT_x<0)
 			{
 				//update bias
 				w[0]=w[0]+trueLabel*LearningRate;
 				
 				//update all other weights

 				for(int j=1;j<w.length;j++)
 				{
 					if(x.containsKey(j))
 							w[j]=w[j]+LearningRate*trueLabel*x.get(j);
 				//	else
 			//			w[j]=(double)0;
 				}
 			}
 			
 		}
 
 		return w;
 	}
 	
 	
 	public static Double[] learnDecayingPerceptron(Data data1[],int index,double LearningRate,int decreaseRate)
 	{
 		Double w[] = new Double[20];  
 		Double min =-0.01;  //  Set To Your Desired Min Value
	    Double max = 0.01;
 		double smallRandNumber = min + new Random().nextDouble() * (max - min); 
 		Arrays.fill(w, smallRandNumber);
 		
 		Data data[] = Arrays.copyOf(data1,index);
 		
 		//define rate of change and pick one randomly
 //		double LearningRates[] = {1, 0.1, 0.01};
 //		int minInt =0;  //  Set To Your Desired Min Value
//	    int maxInt =2;
// 		Random rand = new Random();
 		
// 	 	int randomNum = rand.nextInt((maxInt - minInt) + 1) + minInt;
 	
 	// 	double LearningRate = LearningRates[randomNum];
 	//	double LearningRate = (double)0.1;
 		//check the sign, if -ve then update vector 'w' and bias 'b' as needed.
 	//	int k=0;
 	//	int seed=2500;
 	//	while(k++<seed)
 		
 		
 		//shuffle the data..
 	 	ArrayList<Data> arrayList = new ArrayList<Data>(Arrays.asList(Arrays.copyOf(data,index))); 
 		Collections.shuffle(arrayList);
 		data = arrayList.toArray(new Data[arrayList.size()]);
 	  		
 		for(int i=0;i<index;i++)
 		{
 			
 	//		int minInt1 =0;  //  Set To Your Desired Min Value
 	//	    int maxInt1 =index-1;
 		//    int i=0;
 	 	//	Random rand1= new Random();
 	 		
 	 //	 	i = rand1.nextInt((maxInt1 - minInt1) + 1) + minInt;
 			
 			//Decaying the learning rate
 			LearningRate = LearningRate/(1+decreaseRate);
 			decreaseRate++;
 			
 			double trueLabel;
 			double predictedLabel;
 			double dotProduct_wT_x = 0.0f;
 			HashMap<Integer, Double> x = new HashMap<Integer, Double>();
 		 
 			trueLabel = data[i].getY();
 			x= data[i].getX();
 			
 			//wT*x (contains bias)
 			for (Map.Entry<Integer, Double> entry : x.entrySet()) {
 				dotProduct_wT_x = dotProduct_wT_x +  w[entry.getKey()] * entry.getValue();
 			}
 			
 			//predict the label
 			if(dotProduct_wT_x < 0)
 				predictedLabel = -1;
 			else	
 				predictedLabel = +1;
 			
 			//check if prediction is correct, if not update the weight vector and bias.
 			if(predictedLabel != trueLabel)
 		//	if(dotProduct_wT_x<0)
 			{
 				//update bias
 				w[0]=w[0]+trueLabel*LearningRate;
 				
 				//update all other weights

 				for(int j=1;j<w.length;j++)
 				{
 					if(x.containsKey(j))
 							w[j]=w[j]+LearningRate*trueLabel*x.get(j);
 				//	else
 			//			w[j]=(double)0;
 				}
 			}
 			
 		}
 
 		return w;
 	}
 	
 	
 	public static Double testTest(Double w[])
 	{
 		Data data[]=new Data[1000];
 		int index=-1;
		File file = new File("C:\\Machine Learning\\hw2\\dataset\\diabetes.test");
		 try {
		      FileInputStream fis = new FileInputStream(file);
		      BufferedReader df    = new BufferedReader(new InputStreamReader(fis));
		      String line;
  		
 		      int y = 0 ;
		      HashMap<Integer, Double> x = new HashMap<Integer, Double>();
		   
		      
		      // read the training set and create a Dataset
		      
		      	while ((line = df.readLine())!=null){
 		      			String[] result = line.split("\\ ");
  		 		       x = new HashMap<Integer, Double>();

		      				for (int i=0; i<result.length; i++) {
		      						
		      						if(i==0) //it's a label, create extra 'x' for bias.
		      						{
		      							 y =  Integer.valueOf(result[i]);
		      							 x.put(0,(double)1);
		      						}
		      						else
		      						{
		      							String[] result1 = result[i].split("\\:");
		      							Integer key = Integer.valueOf(result1[0]);
		      							Double value = Double.valueOf(result1[1]);
		      							x.put(key,value);	
		      						}
		      				}
		      			data[++index] = new Data(x,y);	
	 		    }
		      	df.close();
			      
				 //     	
				      	
				      	
				  //TEST
				      	
		 		   }	catch (IOException err) {
		 			   		err.printStackTrace();
				    	}
			return test(data,index,w);

 	}
 	
 	public static Double testDev(Double w[])
 	{
 		Data data[]=new Data[1000];
 		int index=-1;
		File file = new File("C:\\Machine Learning\\hw2\\dataset\\diabetes.dev");
		 try {
		      FileInputStream fis = new FileInputStream(file);
		      BufferedReader df    = new BufferedReader(new InputStreamReader(fis));
		      String line;
  		
 		      int y = 0 ;
		      HashMap<Integer, Double> x = new HashMap<Integer, Double>();
		   
		      
		      // read the training set and create a Dataset
		      
		      	while ((line = df.readLine())!=null){  
 		      			String[] result = line.split("\\ ");
  		 		       x = new HashMap<Integer, Double>();

		      				for (int i=0; i<result.length; i++) {
		      						
		      						if(i==0) //it's a label, create extra 'x' for bias.
		      						{
		      							 y =  Integer.valueOf(result[i]);
		      							 x.put(0,(double)1);
		      						}
		      						else
		      						{
		      							String[] result1 = result[i].split("\\:");
		      							Integer key = Integer.valueOf(result1[0]);
		      							Double value = Double.valueOf(result1[1]);
		      							x.put(key,value);	
		      						}
		      				}
		      			data[++index] = new Data(x,y);	
	 		    }
		      	df.close();
			      
				 //     	
				      	
				      	
				  //TEST
				      	
		 		   }	catch (IOException err) {
		 			   		err.printStackTrace();
				    	}
			return test(data,index,w);

 	}
 	
 	public static Double test(Data data[],int index, Double w[])
 	{
 		int correct=0;
 		int incorrect=0;
 		double accuracy;
 		for(int i=0;i<=index;i++)
 		{
 			double trueLabel;
 			double predictedLabel;
 			double dotProduct_wT_x = 0.0f;
 			HashMap<Integer, Double> x = new HashMap<Integer, Double>();
 			
 			trueLabel = data[i].getY();
 			x= data[i].getX();
 			
 			//wT*x (contains bias)
 			for (Map.Entry<Integer, Double> entry : x.entrySet()) {
 				dotProduct_wT_x = dotProduct_wT_x +  w[entry.getKey()] * entry.getValue();
 			}
 			
 			//predict the label
 			if(dotProduct_wT_x < 0)
 				predictedLabel = -1;
 			else	
 				predictedLabel = +1;
 			
 			//check if prediction is correct, if not update the weight vector and bias.
 			if(predictedLabel == trueLabel)
 	 			correct++;
 			else
 				incorrect++;
 		
 		}
			accuracy =  correct*100/(correct+incorrect);
			return accuracy;
 		
  	}
 	
public static void runSimplePerceptron(Data[] data, int index,Double w[])
{
	 
	 double LearningRates[] = {1, 0.1, 0.01};
		int minInt =0;  //  Set To Your Desired Min Value
	    int maxInt =2;
		Random rand = new Random();
		
	 	int randomNum = rand.nextInt((maxInt - minInt) + 1) + minInt;
	
	 	double LearningRate = LearningRates[randomNum];
	 	
for(int i=0;i<LearningRates.length;i++)
{
	 LearningRate = LearningRates[i];
	 double devAccuracyTotal = (double)0;
	 int noOfEpochs= 20;
	 for(int epoch=0;epoch<noOfEpochs;epoch++)
	 {
	      	w = learnSimplePerceptron(data,index,LearningRate);
	  //    	for(int i=0;i<w.length;i++)
	  //    			System.out.println(w[i]);
// print data     	
//	      	for (Map.Entry<Integer, Double> entry : x.entrySet()) {
//	      System.out.println(""+entry.getKey() + " " + entry.getValue());
//		      	}
	    	      	
     	double devAccuracy = testDev(w);
     	
     	devAccuracyTotal = devAccuracyTotal + devAccuracy;
     	System.out.println("Epoch" + epoch +" Dev Accuracy for Simple Learning Rate " + LearningRate + " is " + devAccuracy);
	 }
	 
	 System.out.println("** Average Dev Accuracy for Simple Learning Rate " + LearningRate + " is " + devAccuracyTotal/noOfEpochs);
	 System.out.println(" ");
}

	double testAccuracy = testTest(w);
  	System.out.println("test Accuracy" + testAccuracy);
}

 	
// Decaying the learning rate
	
public static void runDecayingRatePerceptron(Data[] data, int index,Double w[])
{
 int t=0;
 double LearningRates[] = {1, 0.1, 0.01};
	int minInt =0;  //  Set To Your Desired Min Value
    int maxInt =2;
	Random rand = new Random();
	
 	int randomNum = rand.nextInt((maxInt - minInt) + 1) + minInt;

 	double LearningRate = LearningRates[randomNum];
 	
 	
for(int i=0;i<LearningRates.length;i++)
{
 LearningRate = LearningRates[i];
 double devAccuracyTotal = (double)0;
 int noOfEpochs= 20;
 int decreaseRate = 0;
 for(int epoch=0;epoch<noOfEpochs;epoch++)
 {
	 	w = learnDecayingPerceptron(data,index,LearningRate,decreaseRate);
  //    	for(int i=0;i<w.length;i++)
  //    			System.out.println(w[i]);
//print data     	
//      	for (Map.Entry<Integer, Double> entry : x.entrySet()) {
//      System.out.println(""+entry.getKey() + " " + entry.getValue());
//	      	}
    	      	
	//keep decreaseRate increasing across epochs, by increasing 1 for each example.
	 	decreaseRate = decreaseRate + index;
 	double devAccuracy = testDev(w);
 	
 	devAccuracyTotal = devAccuracyTotal + devAccuracy;
 	System.out.println("Epoch" + epoch +" Dev Accuracy for Decaying Learning Rate " + LearningRate + " is " + devAccuracy);
 }
 
 System.out.println("** Average Dev Accuracy for Decaying Learning Rate " + LearningRate + " is " + devAccuracyTotal/noOfEpochs);
 System.out.println(" ");
}

double testAccuracy = testTest(w);
	System.out.println("test Accuracy" + testAccuracy);
}


}


	