package q3;

public class Garden {
// you are free to add members


	public Garden(){
		// implement your constructor here
	}
	public void startDigging() throws InterruptedException{
        // implement your startDigging method here
	}
	public void doneDigging(){
        // implement your doneDigging method here
	} 
	public void startSeeding() throws InterruptedException{
        // implement your startSeeding method here
	}
	public void doneSeeding(){
        // implement your doneSeeding method here
	} 
	public void startFilling() throws InterruptedException{
        // implement your startFilling method here
	}
	public void doneFilling(){
        // implement your doneFilling method here
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
		    while (true) {
                garden.startDigging();
			    dig();
				garden.doneDigging();
			}
		} 
		
		private void dig(){
		}
	}
	
	protected static class Benjamin implements Runnable {
		Garden garden;
		public Benjamin(Garden garden){
			this.garden = garden;
		}
		@Override
		public void run() {
		    while (true) {
                garden.startSeeding();
				plantSeed();
				garden.doneSeeding();
			}
		} 
		
		private void plantSeed(){
		}
	}
	
	protected static class Mary implements Runnable {
		Garden garden;
		public Mary(Garden garden){
            this.garden = garden;
		}
		@Override
		public void run() {
		    while (true) {
                garden.startFilling();
			 	Fill();
			 	garden.doneFilling();
			}
		} 
		
		private void Fill(){
		}
	}

}
