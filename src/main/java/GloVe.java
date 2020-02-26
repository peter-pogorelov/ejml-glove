import org.ejml.data.DMatrixSparse;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.simple.SimpleMatrix;
import java.util.function.DoubleUnaryOperator;
import java.util.Random;


public class GloVe {
    private SimpleMatrix X;
    private SimpleMatrix W, W_c;
    private double lambda;

    private double maxX;
    private int n_iter;

    public static double randDouble(float min, float max) {
        Random rand = new Random();
        return rand.nextFloat() * (max - min) + min;
    }

    private static void initializeRandomly(SimpleMatrix matrix){
        for(int i = 0; i < matrix.numRows(); ++i){
            for(int j = 0; j < matrix.numCols(); ++j){
                matrix.set(i, j, GloVe.randDouble(-1f, 1f));
            }
        }
    }

    private static double clip(double value){
        return Math.max(Math.min(value, 10), -10);
    }

    private SimpleMatrix toDense(DMatrixSparseCSC m){
        SimpleMatrix result = new SimpleMatrix(m.getNumRows(), m.getNumCols());

        for(int i = 0; i < result.numRows(); i++){
            for(int j = 0; j < result.numCols(); ++j){
                double val = m.get(i, j);

                if(val != 0) {
                    result.set(i, j, m.get(i, j));
                } else{
                    result.set(i, j, 1e-7);
                }
            }
        }

        return result;
    }

    public GloVe(DMatrixSparseCSC cooc_matrix, int embedding_size, int n_iter, double lambda){
        this.n_iter = n_iter;
        this.lambda = lambda;

        this.X = this.toDense(cooc_matrix);
        this.maxX = 100;

        this.W = new SimpleMatrix(this.X.numRows(), embedding_size);
        this.W_c = new SimpleMatrix(this.X.numCols(), embedding_size);

        GloVe.initializeRandomly(this.W);
        GloVe.initializeRandomly(this.W_c);
    }

    private SimpleMatrix applyFunctionTo(SimpleMatrix m, int index, boolean row, DoubleUnaryOperator op){
        if(row) {
            SimpleMatrix r = new SimpleMatrix(1, m.numRows());
            for (int i = 0; i < m.numRows(); i++)
                r.set(0, i, op.applyAsDouble(m.get(index, i)));

            return r;
        } else {
            // mirrored thing, can be optimized
            SimpleMatrix r = new SimpleMatrix(m.numRows(), 1);

            for (int i = 0; i < m.numRows(); i++)
                r.set(i, 0, op.applyAsDouble(m.get(i, index)));

            return r;
        }
    }

    private SimpleMatrix diagonalize(SimpleMatrix row_vector){
        SimpleMatrix matr = new SimpleMatrix(row_vector.numCols(),row_vector.numCols());
        for(int i = 0; i < row_vector.numCols(); ++i)
            matr.set(i, i, row_vector.get(0, i));
        return matr;
    }

    private SimpleMatrix gradW_row(int ix){
        SimpleMatrix w = this.W.rows(ix, ix+1);

        SimpleMatrix c1 = this.applyFunctionTo(this.X, ix, true, (a) -> a < 1e-8 ? -16 : Math.log(a));
        SimpleMatrix c2 = this.applyFunctionTo(this.X, ix, true, (a) -> Math.pow(a / this.maxX, 3./4));
        SimpleMatrix d1 = this.diagonalize(c1);

        SimpleMatrix left = w.mult(this.W_c.transpose().mult(d1).mult(this.W_c));
        SimpleMatrix right = c1.elementMult(c2).mult(this.W_c);

        return left.minus(right);//this.lambda.elementMult(left.minus(right));
    }

    private SimpleMatrix gradWc_row(int ix){
        SimpleMatrix w_c = this.W_c.rows(ix, ix+1);

        SimpleMatrix c1 = this.applyFunctionTo(this.X, ix, false, (a) -> a < 1e-8 ? -16 : Math.log(a)).transpose();
        SimpleMatrix c2 = this.applyFunctionTo(this.X, ix, false, (a) -> Math.pow(a / this.maxX, 3./4)).transpose();

        SimpleMatrix d1 = this.diagonalize(c1);

        SimpleMatrix left = w_c.mult(this.W.transpose().mult(d1).mult(this.W));
        SimpleMatrix right = c1.elementMult(c2).mult(this.W);

        return left.minus(right);//this.lambda.elementMult(left.minus(right));
    }

    private void grad_step(SimpleMatrix m, int index, SimpleMatrix dvec){
        for(int i = 0; i < m.numCols(); ++i){
            m.set(index, i, m.get(index, i) + this.lambda * GloVe.clip(dvec.get(0, i)));
        }
    }

    private double loss(){
        return this.X.divide(this.maxX).elementPower(3./4).elementMult(
                this.W.mult(this.W_c.transpose()).minus(this.X.elementLog()).elementPower(2)
        ).elementSum();
    }

    public void train(){
        System.out.printf("Number of words: %d\n", this.X.numCols());

        for(int i = 0; i < this.n_iter; ++i){
            System.out.println(this.loss());
            //this.W.rows(0, 1).print();

            for(int j = 0; j < this.X.numCols(); ++j){
                this.grad_step(this.W, j, this.gradW_row(j));
            }

            for(int j = 0; j < this.X.numCols(); ++j) {
                this.grad_step(this.W_c, j, this.gradWc_row(j));
            }
        }
    }
}
