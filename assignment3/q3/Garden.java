package q3;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class Garden {
// you are free to add members
int max;
ReentrantLock gardenLock;
ReentrantLock digLock, seedLock, shovelLock;
Condition emptyHole, seededHole, holesAhead;
AtomicInteger holesDug, holesSeeded, holesFilled;
int numPlants;
boolean shovelTaken;

	public Garden(){
		// implement your constructor here
		gardenLock = new ReentrantLock();
		shovelLock = new ReentrantLock();
		seedLock = new ReentrantLock();
		emptyHole = seedLock.newCondition();
		seededHole = shovelLock.newCondition();
		holesAhead = shovelLock.newCondition();
		holesDug = new AtomicInteger();
		holesSeeded = new AtomicInteger();
		holesFilled = new AtomicInteger();
	}
	
	public Garden(int numPlants, int max)
	{
		this();
		this.numPlants = numPlants;
		this.max = max;
	}
	public void startDigging() throws InterruptedException{
        // implement your startDigging method here
		shovelLock.lock();
		while(holesDug.get() - holesFilled.get() > max)
		{
			holesAhead.await();
		}
	}
	public void doneDigging(){
        // implement your doneDigging method here
		boolean signal = false;
		signal = holesDug.incrementAndGet() > holesSeeded.get();
		shovelLock.unlock();
		if(signal)
		{
			seedLock.lock();
			try
			{
				emptyHole.signal();
			}catch (Exception e)
			{
				
			}finally
			{
				seedLock.unlock();
			}
		}
	} 
	public void startSeeding() throws InterruptedException{
        // implement your startSeeding method here
		seedLock.lock();
		while(holesDug.get() <= holesSeeded.get())
		{
			emptyHole.await();
		}
		
	}
	public void doneSeeding(){
        // implement your doneSeeding method here
		boolean signal = holesSeeded.incrementAndGet() > holesFilled.get();
		seedLock.unlock();
		if(signal)
		{
			shovelLock.lock();
			try
			{
				seededHole.signal();
			}catch (Exception e){System.out.println("error in seeding");}
			finally
			{
				shovelLock.unlock();
			}
		}
	} 
	public void startFilling() throws InterruptedException{
        // implement your startFilling method here
		shovelLock.lock();
		while(holesSeeded.get() <= holesFilled.get())
		{
			seededHole.await();
		}
	}
	public void doneFilling(){
        // implement your doneFilling method here
		holesFilled.incrementAndGet();
		holesAhead.signal();
		shovelLock.unlock();
	}

// You are free to implements your own Newton, Benjamin and Mary
// classes. They will NOT count to your grade.
	protected static class Newton implements Runnable {
		Garden garden;
		public Newton(Garden garden){
			this.garden = garden;
		}
		@Override
		public void run() {
		    while (garden.holesDug.get() < garden.numPlants) {
		    	try
		    	{
		    		garden.startDigging();
		    		dig();
		    	} catch (Exception e)
		    	{
		    		System.out.println(e);
		    	}finally
		    	{
		    		garden.doneDigging();
		    	}
                
			    
				
			}
		} 
		
		private void dig(){
			System.out.println("digging: " + garden);
		}
	}
	
	protected static class Benjamin implements Runnable {
		Garden garden;
		public Benjamin(Garden garden){
			this.garden = garden;
		}
		@Override
		public void run() {
		    while (garden.holesSeeded.get() < garden.numPlants) {
		    	try 
		    	{
		    		garden.startSeeding();
		    		plantSeed();
		    	} catch (Exception e)
		    	{
		    		System.out.println(e);
		    	}finally
		    	{
		    		garden.doneSeeding();
		    	}
			}
		} 
		
		private void plantSeed(){
			System.out.println("seeding " + garden);
		}
	}
	
	protected static class Mary implements Runnable {
		Garden garden;
		public Mary(Garden garden){
            this.garden = garden;
		}
		@Override
		public void run() {
		    while (garden.holesFilled.get() < garden.numPlants) {
		    	try
		    	{
		    		garden.startFilling();
		    		Fill();
		    	} catch (Exception e)
		    	{
		    		System.out.println(e);
		    	}finally
		    	{
		    		garden.doneFilling();
		    	}
			}
		} 
		
		private void Fill(){
			System.out.println("filling " + garden);
		}
	}

	public String toString()
	{
		String stateOfGarden = "holes dug: " + holesDug
				+ "\nholes seeded: " + holesSeeded
				+ "\nholes filled: " + holesFilled
				+ "\nshovel taken?: " + shovelTaken + "\n";
		return stateOfGarden;
	}
	
	public static void main(String args[])
	{
		Garden g = new Garden(10, 2);
		Thread mary = new Thread(new Mary(g));
		Thread ben = new Thread(new Benjamin(g));
		Thread newton = new Thread(new Newton(g));
		mary.start();
		ben.start();
		newton.start();
	}
}
