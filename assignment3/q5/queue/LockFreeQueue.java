package q5;//ueue;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class LockFreeQueue implements MyQueue {
// you are free to add members
AtomicInteger count;
AtomicReference<SmartPointer> head;
AtomicReference<SmartPointer> tail;

  public LockFreeQueue() {
	// implement your constructor here
	  SmartPointer sentinel = new SmartPointer(new Node(0), 0);
	  head = new AtomicReference<SmartPointer>(sentinel);
	  tail = new AtomicReference<SmartPointer>(sentinel);
	  count = new AtomicInteger();
  }

  public boolean enq(Integer value) {
	// implement your enq method here
	  Node nodeToEnq = new Node(value);
	  AtomicReference<SmartPointer> tempTail = new AtomicReference<SmartPointer>(null);
	  while(true)
	  {
		  tempTail = new AtomicReference<SmartPointer>(tail.get());
		  AtomicReference<SmartPointer> next = new AtomicReference<SmartPointer>(tempTail.get().ptr.get().next.get());
		  if(tail.get() == tempTail.get())
		  {
			  if(next.get().ptr.get() == null)
			  {
				  if(tempTail.get().ptr.get().next.compareAndSet(next.get(), new SmartPointer(nodeToEnq, next.get().cnt.get() + 1)))
				  {
					  break;
				  }
			  }else
			  {
				  tail.compareAndSet(tempTail.get(), new SmartPointer(next.get().ptr.get(), tempTail.get().cnt.get() + 1));
			  }
		  }
	  }
    tail.compareAndSet(tempTail.get(), new SmartPointer(nodeToEnq, tempTail.get().cnt.get() + 1));
    count.incrementAndGet();
    return true;
  }
  
  public Integer deq() {
	// implement your deq method here
	Integer pvalue = null;
    while(true)
    {
    	AtomicReference<SmartPointer> tempTail = new AtomicReference<SmartPointer>(tail.get());
    	AtomicReference<SmartPointer> tempHead = new AtomicReference<SmartPointer>(head.get());
    	AtomicReference<SmartPointer> next = new AtomicReference<SmartPointer>(tempHead.get().ptr.get().next.get());
    	if(tempHead.get() == head.get())
    	{
    		if(tempHead.get().ptr.get() == tempTail.get().ptr.get())
    		{
    			if(next.get().ptr.get() == null)
    			{
    				return null;
    			}
    			tail.compareAndSet(tempTail.get(), new SmartPointer(next.get().ptr.get(), tempTail.get().cnt.get() + 1));
    		} else
    		{
    			pvalue = next.get().ptr.get().value;
    			if(head.compareAndSet(tempHead.get(), new SmartPointer(next.get().ptr.get(), tempHead.get().cnt.get() + 1)))
    			{
    				break;
    			}
    		}
    	}
    }
    count.decrementAndGet();
    return pvalue;
  }
  
  public boolean contains(Integer x)
  {
	  AtomicReference<SmartPointer> sp = head.get().ptr.get().next;
	  SmartPointer next = sp.get();
	  while(next.ptr.get() != null)
	  {
		  if(next.ptr.get().value == x)
		  {
			  return true;
		  }
		  while(true)
		  {
			  if(sp.compareAndSet(next, new SmartPointer(next.ptr.get().next.get().ptr.get(), sp.get().cnt.get() + 1)))
			  {
				  break;
			  }
		  }
		  
		  next = sp.get();
	  }
	  return false;
  }
  
  protected class SmartPointer {
	  AtomicReference<Node> ptr;
	  AtomicInteger cnt;
	  public SmartPointer(Node node) {
		  this(node, 0);
	  }
	  public SmartPointer(Node node, int count) {
		  ptr = new AtomicReference<Node>(node);
		  this.cnt = new AtomicInteger(count);
	  }
  }
  
  protected class Node {
	  public Integer value;
	  public AtomicReference<SmartPointer> next;
	  
	  public Node(Integer x) {
		  value = x;
		  next = new AtomicReference<SmartPointer>(new SmartPointer(null, 0));
	  }
  }
}
