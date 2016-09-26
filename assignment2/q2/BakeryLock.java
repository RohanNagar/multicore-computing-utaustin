package q2;

import java.util.concurrent.atomic.AtomicIntegerArray;

// TODO
// Implement the bakery algorithm

public class BakeryLock implements MyLock {
	AtomicIntegerArray numbers;
	AtomicIntegerArray choosing; //{0 = false, not 0 = true} 
	int numThread;
    public BakeryLock(int numThread) {
    	numbers = new AtomicIntegerArray(numThread);
    	choosing = new AtomicIntegerArray(numThread);
    	this.numThread = numThread;
    }

    //assumes myId is in [0, numThread]
    @Override
    public void lock(int myId) {
    	//phase 1
    	try{
    		choosing.set(myId, 1);
    	}catch(IndexOutOfBoundsException e)
    	{
    		System.err.printf("%d %d", choosing.length(), myId);
    	}
    	int max = numbers.get(0);
    	for(int i = 1; i < numbers.length(); i++)
    	{
    		int number = numbers.get(i);
    		if(number > max)
    		{
    			max = number; 
    		}
    	}
    	numbers.set(myId, max+1);
    	choosing.set(myId, 0);
    	
    	//phase 2
    	for(int i = 0; i < numThread; i++)
    	{
    		while(choosing.get(i) != 0){};
    		while((numbers.get(i) != 0) && ((numbers.get(i) < numbers.get(myId)) || ((numbers.get(i) == numbers.get(myId)) && i < myId))){};
    	}
    }

    @Override
    public void unlock(int myId) {
    	numbers.set(myId, 0);
    }
}
