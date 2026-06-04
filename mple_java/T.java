import org.javatuples.*;
class T { public static void main(String[] a){ Pair<Integer,Integer> p = Pair.with(1,2); assert(p.getValue0()==1); System.out.println("ok"); } }
