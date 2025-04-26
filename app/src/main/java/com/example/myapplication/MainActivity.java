package com.example.myapplication;

import java.util.Arrays;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;


//Pytorch
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

//오디오 input
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

//Android 입출력
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

//CHeck log
import android.util.Log;

//Connecting UI
import android.view.View;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;



public class MainActivity extends AppCompatActivity {

    private Module model;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_userinfo);








    }


}