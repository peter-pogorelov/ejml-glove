import org.ejml.data.DMatrixSparseCSC;
import org.ejml.simple.SimpleMatrix;

import java.io.*;
import java.util.List;
import java.util.ArrayList;

public class Main{

    private static String loadFile(File file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        StringBuilder sb = new StringBuilder();
        try{
            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            br.close();
            return sb.toString();
        }
    }

    private static List<String> loadTextsFromFolder(String path) throws IOException {
        List<String> result = new ArrayList<String>();

        File folder = new File(path);
        File[] listOfFiles = folder.listFiles();

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                result.add(Main.loadFile(listOfFiles[i]));
            }
        }

        return result;
    }

    public static void main(String[] args) throws IOException {
        OccurenceMatrixBuilder builder = new OccurenceMatrixBuilder(2);
        //String text = Main.loadFile(new File("<folder>"));

        String[] texts = new String[]{
                "\"From: lerxst@wam.umd.edu (where's my thing)\\nSubject: WHAT car is this!?\\nNntp-Posting-Host: rac3.wam.umd.edu\\nOrganization: University of Maryland, College Park\\nLines: 15\\n\\n I was wondering if anyone out there could enlighten me on this car I saw\\nthe other day. It was a 2-door sports car, looked to be from the late 60s/\\nearly 70s. It was called a Bricklin. The doors were really small. In addition,\\nthe front bumper was separate from the rest of the body. This is \\nall I know. If anyone can tellme a model name, engine specs, years\\nof production, where this car is made, history, or whatever info you\\nhave on this funky looking car, please e-mail.\\n\\nThanks,\\n- IL\\n   ---- brought to you by your neighborhood Lerxst ----\\n\\n\\n\\n\\n\"",
                "From: guykuo@carson.u.washington.edu (Guy Kuo)\\nSubject: SI Clock Poll - Final Call\\nSummary: Final call for SI clock reports\\nKeywords: SI,acceleration,clock,upgrade\\nArticle-I.D.: shelley.1qvfo9INNc3s\\nOrganization: University of Washington\\nLines: 11\\nNNTP-Posting-Host: carson.u.washington.edu\\n\\nA fair number of brave souls who upgraded their SI clock oscillator have\\nshared their experiences for this poll. Please send a brief message detailing\\nyour experiences with the procedure. Top speed attained, CPU rated speed,\\nadd on cards and adapters, heat sinks, hour of usage per day, floppy disk\\nfunctionality with 800 and 1.4 m floppies are especially requested.\\n\\nI will be summarizing in the next two days, so please add to the network\\nknowledge base if you have done the clock upgrade and haven't answered this\\npoll. Thanks.\\n\\nGuy Kuo <guykuo@u.washington.edu>\\n"
        };

        System.out.println("Building word index...");
        builder.buildWordMap(texts);
        //System.out.println(builder.word2index);
        System.out.println("Building cooccurence matrix...");
        DMatrixSparseCSC cooc_matrix = builder.buildMatrixCSC(texts);
        System.out.println("Training embeddings... ");
        GloVe glove = new GloVe(cooc_matrix, 50, 800, 0.01);
        glove.train();
    }
}
