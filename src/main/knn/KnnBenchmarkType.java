package knn;

/** Indicates the type of Knn Benchmark being profiled.
 * Used to create the appropriate index structure for each type.
 */
public enum KnnBenchmarkType {

  /** Each vector indexed as a separate document */
  DEFAULT(""),

  /** Maintains parent-child relation between vectors */
  PARENT_JOIN("parentJoin"),

  /** Indexes multi-vector values in the document */
  MULTI_VECTOR("multiVector");

  /** Added to index path for identification */
  public final String indexTag;

  KnnBenchmarkType(String tag) {
    indexTag = tag;
  }
}
