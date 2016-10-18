package q4;

import java.util.concurrent.locks.ReentrantLock;

public class CoarseGrainedListSet implements ListSet {
  private final ReentrantLock lock = new ReentrantLock();
  private Node head;
	
  public CoarseGrainedListSet() {
    head = new Node(null);
  }
  
  public boolean add(int value) {
    Node newNode = new Node(value);
    Node current = head;

    lock.lock();
    while (current.next != null) {
      if (value <= current.next.value) {
        break;
      }

      current = current.next;
    }

    newNode.next = current.next;
    current.next = newNode;
    lock.unlock();

    return true;
  }
  
  public boolean remove(int value) {
    Node current = head;
    Node prev = head;

    lock.lock();
    while (current != null) {
      if (current.value != null && current.value == value) {
        prev.next = current.next;

        lock.unlock();
        return true;
      }

      prev = current;
      current = current.next;
    }

    lock.unlock();
    return false;
  }
  
  public boolean contains(int value) {
    Node current = head;

    lock.lock();
    while (current != null) {
      if (current.value != null && current.value == value) {
        lock.unlock();
        return true;
      }

      if (current.value != null && current.value > value) {
        lock.unlock();
        return false;
      }

      current = current.next;
    }

    lock.unlock();
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

//  public static void main(String[] args) {
//    CoarseGrainedListSet list = new CoarseGrainedListSet();
//
//    Thread t1 = new Thread() {
//      @Override
//      public void run() {
//        list.add(3);
//        list.add(7);
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
//        System.out.println(list.contains(18));
//      }
//    };
//
//    t1.start();
//    t2.start();
//  }
}
