package knn;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.util.List;

import static knn.KnnGraphTester.KNN_FIELD;

public class KnnFloatVectorBenchmarkQuery extends KnnFloatVectorQuery {


  public KnnFloatVectorBenchmarkQuery(float[] target, int k) {
    super(KNN_FIELD, target, k);
  }

  public KnnFloatVectorBenchmarkQuery(float[] target, int k, Query filter) {
    super(KNN_FIELD, target, k, filter);
  }

  @Override
  public TopDocs exactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout) throws IOException {
    return super.exactSearch(context, acceptIterator, queryTimeout);
  }

  // TODO: refactor to a util and make the query return acceptIterator
  public static TopDocs runExactSearch(IndexReader reader, KnnFloatVectorBenchmarkQuery query) throws IOException {
    List<LeafReaderContext> leafReaderContexts = reader.leaves();
    TopDocs[] perLeafResults = new TopDocs[leafReaderContexts.size()];
    int leaf = 0;
    for (LeafReaderContext ctx : leafReaderContexts) {
      Bits liveDocs = ctx.reader().getLiveDocs();
      FilteredDocIdSetIterator acceptDocs =
        new FilteredDocIdSetIterator(DocIdSetIterator.all(ctx.reader().maxDoc())) {
          @Override
          protected boolean match(int doc) {
            return liveDocs == null || liveDocs.get(doc);
          }
      };
      perLeafResults[leaf] = query.exactSearch(ctx, acceptDocs, null);
      if (ctx.docBase > 0) {
        for (ScoreDoc scoreDoc : perLeafResults[leaf].scoreDocs) {
          scoreDoc.doc += ctx.docBase;
        }
      }
      leaf++;
    }
    return query.mergeLeafResults(perLeafResults);
  }
}
