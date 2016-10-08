package q4;

public class CoarseGrainedListSet implements ListSet {
// you are free to add members
	
  public CoarseGrainedListSet() {
	// implement your constructor here	  
  }
  
  public boolean add(int value) {
	// implement your add method here	
    return false;
  }
  
  public boolean remove(int value) {
	// implement your remove method here	
    return false;
  }
  
  public boolean contains(int value) {
	// implement your contains method here	
    return false;
  }
  
  protected class Node {
	  public Integer value;
	  public Node next;
		    
	  public Node(Integer x) {
		  value = x;
		  next = null;
	  }
  }
}
