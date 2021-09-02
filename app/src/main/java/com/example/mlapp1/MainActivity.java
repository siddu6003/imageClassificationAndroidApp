package com.example.mlapp1;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.mlapp1.ml.MobilenetV110224Quant;

import org.checkerframework.checker.units.qual.A;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class MainActivity extends AppCompatActivity {
    private Button b1;
    private Button b2;
    private Button b3;
    String filename="labels.txt";
    Bitmap b;
    TextView t1;
    ImageView iv;
    String[] l=new String[1001];
    List<String> x=new ArrayList<>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        b1=(Button) findViewById(R.id.select);
        b3=(Button) findViewById(R.id.predict);
        t1=(TextView) findViewById(R.id.textView);
        iv=(ImageView)findViewById(R.id.imageView);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt"))); //throwing a FileNotFoundException?
            String word;
            while((word=br.readLine()) != null)
               x.add(word); //break txt file into different words, add to wordList
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        finally {
            try {
                br.close(); //stop reading
            }
            catch(IOException ex) {
                ex.printStackTrace();
            }
        }
        String[] words = new String[x.size()];
        x.toArray(words); //make array of wordList

        b1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
               Intent i= new Intent();
                i.setType("image/*");
                i.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(i,100);

            }
        });
        b3.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {
                Bitmap resized=Bitmap.createScaledBitmap(b,224,224,true);
                try {
                    MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
                    TensorImage t=new TensorImage();
                    t.load(resized);
                    ByteBuffer byteBuffer = t.getBuffer();
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                   float[] a=outputFeature0.getFloatArray();
                   int max=maxIndex(a);

                   t1.setText(String.valueOf(words[max]));
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        iv.setImageURI(data.getData());
        Uri uri = data.getData();
        try {
            b = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static int maxIndex(float[] a) {
        int Index=-1;
        float max=Integer.MIN_VALUE;
        for(int i=0;i<a.length;i++){
            if(a[i]>max){
                max=a[i];
                Index=i;
            }
        }
        return Index;
    }
}