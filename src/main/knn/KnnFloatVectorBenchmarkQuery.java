package knn;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

import java.io.IOException;
import java.util.List;

public class KnnFloatVectorBenchmarkQuery extends KnnFloatVectorQuery {


  public KnnFloatVectorBenchmarkQuery(String field, float[] target, int k) {
    super(field, target, k);
  }

  public KnnFloatVectorBenchmarkQuery(String field, float[] target, int k, Query filter) {
    super(field, target, k, filter);
  }

  @Override
  public TopDocs exactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout) throws IOException {
    return super.exactSearch(context, acceptIterator, queryTimeout);
  }

  // TODO: refactor to a util and make the query return acceptIterator
  public static TopDocs runExactSearch(IndexReader reader, KnnFloatVectorBenchmarkQuery query) throws IOException {
    IndexSearcher searcher = new IndexSearcher(reader);
    List<LeafReaderContext> leafReaderContexts = reader.leaves();
    TopDocs[] perLeafResults = new TopDocs[leafReaderContexts.size()];
    int leaf = 0;
    for (LeafReaderContext ctx : leafReaderContexts) {

      ctx.reader().getLiveDocs();
    }
  }
}
