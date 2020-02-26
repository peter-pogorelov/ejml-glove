import java.nio.Buffer;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.StringTokenizer;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DMatrixSparse;
import org.ejml.simple.SimpleMatrix;

public class OccurenceMatrixBuilder {
    private int wordCount = 0;
    private int window;
    public Map<String, Integer> word2index;

    public OccurenceMatrixBuilder(int window){
        assert(window > 0);

        this.window = window;
        this.word2index = new HashMap<>();
    }

    private void updateMap(String text){
        for(String token: text.split("\\s")){
            if(!this.word2index.containsKey(token)) {
                this.word2index.put(token, wordCount++);
            }
        }
    }

    public int buildWordMap(String[] texts){
        for(String text: texts){
            this.updateMap(text);
        }
        return this.wordCount;
    }

    public DMatrixSparseCSC buildMatrixCSC(String[] texts){
        DMatrixSparseCSC matrix = new DMatrixSparseCSC(this.wordCount, this.wordCount);

        for(String text: texts){

            String[] listOfStrings = text.split("\\s");

            for(int i = 0; i < listOfStrings.length; ++i){
                int row = this.word2index.get(listOfStrings[i]);

                for(int j = Math.max(i - this.window, 0); j < i; ++j){
                    int column = this.word2index.get(listOfStrings[j]);

                    matrix.set(row, column,matrix.get(row, column) + 1);
                    matrix.set(column, row,matrix.get(column, row) + 1);
                }
            }
        }

        return matrix;
    }

    public SimpleMatrix buildMatrix(String[] texts){
        SimpleMatrix matrix = new SimpleMatrix(this.wordCount, this.wordCount);
        matrix.fill(0);

        for(String text: texts){
            String[] listOfStrings = text.split("\\s");

            for(int i = 0; i < listOfStrings.length; ++i){
                int row = this.word2index.get(listOfStrings[i]);

                for(int j = Math.max(i - this.window, 0); j < i; ++j){
                    int column = this.word2index.get(listOfStrings[j]);

                    matrix.set(row, column,matrix.get(row, column) + 1);
                    matrix.set(column, row,matrix.get(column, row) + 1);
                }
            }
        }

        return matrix;
    }
}
