/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package knn;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.*;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MultiVectorSimilarityFunction;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static knn.KnnGraphTester.*;
import static org.apache.lucene.index.MultiVectorSimilarityFunction.Aggregation.SUM_MAX;

public class KnnIndexer {
  // use smaller ram buffer so we get to merging sooner, making better use of
  // many cores (TODO: use multiple indexing threads):
  // private static final double WRITER_BUFFER_MB = 1994d;
  private static final double WRITER_BUFFER_MB = 64;

  Path docsPath;
  Path indexPath;
  VectorEncoding vectorEncoding;
  int dim;
  VectorSimilarityFunction similarityFunction;
  Codec codec;
  int numDocs;
  int docsStartIndex;
  boolean quiet;
  KnnBenchmarkType benchmarkType;
  Path metadataFilePath;

  public KnnIndexer(Path docsPath, Path indexPath, Codec codec, VectorEncoding vectorEncoding, int dim,
                    VectorSimilarityFunction similarityFunction, int numDocs, int docsStartIndex, boolean quiet,
                    KnnBenchmarkType benchmarkType, Path metadataFile) {
    this.docsPath = docsPath;
    this.indexPath = indexPath;
    this.codec = codec;
    this.vectorEncoding = vectorEncoding;
    this.dim = dim;
    this.similarityFunction = similarityFunction;
    this.numDocs = numDocs;
    this.docsStartIndex = docsStartIndex;
    this.quiet = quiet;
    this.benchmarkType = benchmarkType;
    this.metadataFilePath = metadataFile;
  }

  public int createIndex() throws IOException {
    IndexWriterConfig iwc = new IndexWriterConfig().setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    iwc.setCodec(codec);
    // iwc.setMergePolicy(NoMergePolicy.INSTANCE);
    iwc.setRAMBufferSizeMB(WRITER_BUFFER_MB);
    iwc.setUseCompoundFile(false);
    // iwc.setMaxBufferedDocs(10000);

    FieldType fieldType =
        switch (vectorEncoding) {
          case BYTE -> (benchmarkType == KnnBenchmarkType.MULTI_VECTOR) ?
              KnnByteMultiVectorField.createFieldType(dim, new MultiVectorSimilarityFunction(similarityFunction, SUM_MAX)) :
              KnnByteVectorField.createFieldType(dim, similarityFunction);
          case FLOAT32 -> (benchmarkType == KnnBenchmarkType.MULTI_VECTOR) ?
              KnnFloatMultiVectorField.createFieldType(dim, new MultiVectorSimilarityFunction(similarityFunction, SUM_MAX)) :
              KnnFloatVectorField.createFieldType(dim, similarityFunction);
        };
    if (quiet == false) {
//      iwc.setInfoStream(new PrintStreamInfoStream(System.out));
      System.out.println("creating index in " + indexPath);
    }

    if (!indexPath.toFile().exists()) {
      indexPath.toFile().mkdirs();
    }

    long start = System.nanoTime();
    try (FSDirectory dir = FSDirectory.open(indexPath);
         IndexWriter iw = new IndexWriter(dir, iwc)) {
      try (FileChannel in = FileChannel.open(docsPath)) {
        if (docsStartIndex > 0) {
          seekToStartDoc(in, dim, vectorEncoding, docsStartIndex);
        }
        VectorReader vectorReader = VectorReader.create(in, dim, vectorEncoding);
        log("benchmarkType = %s", benchmarkType);
        switch (benchmarkType) {
          case PARENT_JOIN -> createParentJoinIndex(vectorReader, fieldType, iw);
          case DEFAULT -> createDefaultIndex(vectorReader, fieldType, iw);
        }
      }
    }
    long elapsed = System.nanoTime() - start;
    log("Indexed %d docs in %d seconds", numDocs, TimeUnit.NANOSECONDS.toSeconds(elapsed));
    return (int) TimeUnit.NANOSECONDS.toMillis(elapsed);
  }

  private void createDefaultIndex(VectorReader vectorReader, FieldType fieldType, IndexWriter iw) throws IOException {
    for (int i = 0; i < numDocs; i++) {
      Document doc = new Document();
      switch (vectorEncoding) {
        case BYTE -> doc.add(
          new KnnByteVectorField(
            KnnGraphTester.KNN_FIELD, ((VectorReaderByte) vectorReader).nextBytes(), fieldType));
        case FLOAT32 -> doc.add(
          new KnnFloatVectorField(KnnGraphTester.KNN_FIELD, vectorReader.next(), fieldType));
      }
      doc.add(new StoredField(KnnGraphTester.ID_FIELD, i));
      iw.addDocument(doc);

      if ((i + 1) % 25000 == 0) {
        System.out.println("Done indexing " + (i + 1) + " documents.");
      }
    }
  }

  private void createParentJoinIndex(VectorReader vectorReader, FieldType fieldType, IndexWriter iw) throws IOException {
    try (BufferedReader br = Files.newBufferedReader(metadataFilePath)) {
      String[] headers = br.readLine().trim().split(",");
      if (headers.length != 2) {
        throw new IllegalStateException("Expected two columns in metadata csv. Found: " + headers.length);
      }
      log("Metadata file columns: %s | %s", headers[0], headers[1]);
      int childDocs = 0;
      int parentDocs = 0;
      int docIds = 0;
      String prevWikiId = "null";
      String currWikiId;
      List<Document> block = new ArrayList<>();
      do {
        String[] line = br.readLine().trim().split(",");
        currWikiId = line[0];
        String currParaId = line[1];
        Document doc = new Document();
        switch (vectorEncoding) {
          case BYTE -> doc.add(
            new KnnByteVectorField(
              KnnGraphTester.KNN_FIELD, ((VectorReaderByte) vectorReader).nextBytes(), fieldType));
          case FLOAT32 -> doc.add(
            new KnnFloatVectorField(KnnGraphTester.KNN_FIELD, vectorReader.next(), fieldType));
        }
        doc.add(new StoredField(KnnGraphTester.ID_FIELD, docIds++));
        doc.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, currWikiId, Field.Store.YES));
        doc.add(new StringField(KnnGraphTester.WIKI_PARA_ID_FIELD, currParaId, Field.Store.YES));
        doc.add(new StringField(KnnGraphTester.DOCTYPE_FIELD, DOCTYPE_CHILD, Field.Store.NO));
        childDocs++;

        // Close block and create a new one when wiki article changes.
        if (!currWikiId.equals(prevWikiId) && !"null".equals(prevWikiId)) {
          Document parent = new Document();
          parent.add(new StoredField(KnnGraphTester.ID_FIELD, docIds++));
          parent.add(new StringField(KnnGraphTester.DOCTYPE_FIELD, DOCTYPE_PARENT, Field.Store.NO));
          parent.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, prevWikiId, Field.Store.YES));
          parent.add(new StringField(KnnGraphTester.WIKI_PARA_ID_FIELD, "_", Field.Store.YES));
          block.add(parent);
          iw.addDocuments(block);
          parentDocs++;
          // create new block for the next article
          block = new ArrayList<>();
          block.add(doc);
        } else {
          block.add(doc);
        }
        prevWikiId = currWikiId;
        if (childDocs % 25000 == 0) {
          log("indexed %d child documents, with %d parents", childDocs, parentDocs);
        }
      } while (childDocs < numDocs);
      if (!block.isEmpty()) {
        Document parent = new Document();
        parent.add(new StoredField(KnnGraphTester.ID_FIELD, docIds++));
        parent.add(new StringField(KnnGraphTester.DOCTYPE_FIELD, DOCTYPE_PARENT, Field.Store.NO));
        parent.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, prevWikiId, Field.Store.YES));
        parent.add(new StringField(KnnGraphTester.WIKI_PARA_ID_FIELD, "_", Field.Store.YES));
        block.add(parent);
        iw.addDocuments(block);
      }
      log("Indexed %d documents with %d parent docs. now flush", childDocs, parentDocs);
    }
  }

  private void createMultiVectorIndex(VectorReader vectorReader, FieldType fieldType, IndexWriter iw) throws IOException {
    try (BufferedReader br = Files.newBufferedReader(metadataFilePath)) {
      String[] headers = br.readLine().trim().split(",");
      if (headers.length != 2) {
        throw new IllegalStateException("Expected two columns in metadata csv. Found: " + headers.length);
      }
      log("Metadata file columns: %s | %s", headers[0], headers[1]);
      int docIds = 0;
      int vectorsRead = 0;
      int minVectorsPerDoc = Integer.MAX_VALUE;
      int maxVectorsPerDoc = Integer.MIN_VALUE;
      String prevWikiId = "null";
      String currWikiId;
      String currParaId;
      List<float[]> floatVectorValues = new ArrayList<>();
      List<byte[]> byteVectorValues = new ArrayList<>();
      do {
        String[] line = br.readLine().trim().split(",");
        currWikiId = line[0];
        currParaId = line[1];
        switch (vectorEncoding) {
          case BYTE -> {
            byte[] vector = new byte[dim];
            byte[] inputVal = ((VectorReaderByte) vectorReader).nextBytes();
            System.arraycopy(inputVal, 0, vector, 0, dim);
            byteVectorValues.add(vector);
            vectorsRead++;
          }
          case FLOAT32 -> {
            float[] vector = new float[dim];
            float[] inputVal = vectorReader.next();
            System.arraycopy(inputVal, 0, vector, 0, dim);
            floatVectorValues.add(vector);
            vectorsRead++;
          }
        }

        if (currWikiId.equals(prevWikiId) == false && "null".equals(prevWikiId) == false) {
          Document doc = new Document();
          doc.add(new StoredField(KnnGraphTester.ID_FIELD, docIds++));
          doc.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, currWikiId, Field.Store.YES));
          // TODO: index a multi-vector field
          switch (vectorEncoding) {
            case BYTE -> {
              doc.add(new KnnByteMultiVectorField(KnnGraphTester.KNN_FIELD, byteVectorValues, fieldType));
//              doc.add(new KnnByteVectorField(KnnGraphTester.KNN_FIELD, byteVectorValues.getFirst(), fieldType));
              minVectorsPerDoc = Integer.min(minVectorsPerDoc, byteVectorValues.size());
              maxVectorsPerDoc = Integer.max(maxVectorsPerDoc, byteVectorValues.size());
              byteVectorValues.clear();
            }
            case FLOAT32 -> {
              doc.add(new KnnFloatMultiVectorField(KnnGraphTester.KNN_FIELD, floatVectorValues, fieldType));
//              doc.add(new KnnFloatVectorField(KnnGraphTester.KNN_FIELD, floatVectorValues.getFirst(), fieldType));
              minVectorsPerDoc = Integer.min(minVectorsPerDoc, floatVectorValues.size());
              maxVectorsPerDoc = Integer.max(maxVectorsPerDoc, floatVectorValues.size());
              floatVectorValues.clear();
            }
          }
          iw.addDocument(doc);
          prevWikiId = currWikiId;
          if (docIds % 25_000 == 0) {
            log("\t...documents indexed: %d, vectors indexed: %d, vectors per doc: min=%d, avg=%d, max=%d",
              docIds, vectorsRead, minVectorsPerDoc, vectorsRead / docIds, maxVectorsPerDoc);
            log("\t... vectors in last doc: %d, last paraId: %s, wikiId: %s", floatVectorValues.size(), currParaId, prevWikiId);
          }
        }
      } while (vectorsRead < numDocs);

      // add final chunk of vectors
      if (byteVectorValues.isEmpty() == false || floatVectorValues.isEmpty() == false) {
        Document doc = new Document();
        doc.add(new StoredField(KnnGraphTester.ID_FIELD, docIds++));
        doc.add(new StringField(KnnGraphTester.WIKI_ID_FIELD, currWikiId, Field.Store.YES));
        // TODO: index a multi-vector field
        switch (vectorEncoding) {
          case BYTE -> {
            doc.add(new KnnByteMultiVectorField(KnnGraphTester.KNN_FIELD, byteVectorValues, fieldType));
//            doc.add(new KnnByteVectorField(
//              KnnGraphTester.KNN_FIELD, byteVectorValues.getFirst(), fieldType));
            minVectorsPerDoc = Integer.min(minVectorsPerDoc, byteVectorValues.size());
            maxVectorsPerDoc = Integer.max(maxVectorsPerDoc, byteVectorValues.size());
            byteVectorValues.clear();
          }
          case FLOAT32 -> {
            doc.add(new KnnFloatMultiVectorField(KnnGraphTester.KNN_FIELD, floatVectorValues, fieldType));
//            doc.add(
//              new KnnFloatVectorField(KnnGraphTester.KNN_FIELD, floatVectorValues.getFirst(), fieldType));
            minVectorsPerDoc = Integer.min(minVectorsPerDoc, floatVectorValues.size());
            maxVectorsPerDoc = Integer.max(maxVectorsPerDoc, floatVectorValues.size());
            floatVectorValues.clear();
          }
        }
        iw.addDocument(doc);
      }
      log("Documents indexed: %d, vectors indexed: %d, vectors per doc: min=%d, avg=%d, max=%d",
        docIds, vectorsRead, minVectorsPerDoc, vectorsRead / docIds, maxVectorsPerDoc);
      log("\t... last chunk done. vectors in last doc: %d, last paraId: %s, wikiId: %s", floatVectorValues.size(), currParaId, prevWikiId);
    }
  }

  private void seekToStartDoc(FileChannel in, int dim, VectorEncoding vectorEncoding, int docsStartIndex) throws IOException {
    int startByte = docsStartIndex * dim * vectorEncoding.byteSize;
    in.position(startByte);
  }

  private void log(String msg, Object... args) {
    if (quiet == false) {
      System.out.printf((msg) + "%n", args);
    }
  }
}
