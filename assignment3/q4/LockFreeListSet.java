package q4;

import java.util.concurrent.atomic.AtomicReference;

public class LockFreeListSet implements ListSet {
  private Node head;
	
  public LockFreeListSet() {
    head = new Node(null);
  }
	  
  public boolean add(int value) {
    Node newNode = new Node(value);
    Node current = head;

    while (current.next != null) {
      if (value <= current.next.value) {
        break;
      }

      current = current.next;
    }

    while (true) {
      newNode.next = current.next;
      Node oldNext = current.next;

      AtomicReference<Node> ref = new AtomicReference<>(current.next);
      if (ref.compareAndSet(oldNext, newNode)) {
        current.next = ref.get();
        return true;
      }
    }
  }
	  
  public boolean remove(int value) {
    /** NO IMPLEMENTATION NECESSARY */
	  return false;
  }
	  
  public boolean contains(int value) {
	  Node current = head;

    while (current != null) {
      if (current.value != null && value <= current.value) {
        break;
      }

      current = current.next;
    }

	  return current != null && current.value == value;
  }
	  
  protected class Node {
	  public Integer value;
	  public Node next;
			    
	  public Node(Integer x) {
		  value = x;
		  next = null;
	  }
  }

//  public static void main(String[] args) {
//    LockFreeListSet list = new LockFreeListSet();
//
//    Thread t1 = new Thread() {
//      @Override
//      public void run() {
//        list.add(3);
//        list.add(7);
//        list.add(8);
//        list.add(10);
//        list.add(11);
//        list.add(235);
//
//        list.remove(5);
//
//        System.out.println(list.contains(3));
//      }
//    };
//
//    Thread t2 = new Thread() {
//      @Override
//      public void run() {
//        list.add(16);
//        list.add(5);
//
//        System.out.println(list.contains(5));
//        System.out.println(list.contains(235));
//      }
//    };
//
//    t1.start();
//    t2.start();
//  }
}
