package com.example.myapplication;

import java.util.Arrays;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.view.GravityCompat;
import androidx.drawerlayout.widget.DrawerLayout;


//Pytorch
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

//오디오 input
import android.content.Intent;
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
    private DrawerLayout drawerLayout;
    private Button menuButton, btnKeyword, btnVoiceRegister;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        drawerLayout = findViewById(R.id.drawerLayout);
        menuButton = findViewById(R.id.menuButton);
        btnKeyword = findViewById(R.id.button1); // 키워드 버튼
        btnVoiceRegister = findViewById(R.id.button2); // 사용자 음성 등록 버튼

        // 햄버거 메뉴 클릭 → 메뉴 열기
        menuButton.setOnClickListener(v -> {
            drawerLayout.openDrawer(GravityCompat.END); // 오른쪽 메뉴 열기
        });

        // 키워드 버튼 클릭 → 키워드 등록 화면 이동
        btnKeyword.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, KeywordActivity.class);
            startActivity(intent);
        });

        // 사용자 음성 등록 버튼 클릭 → 음성 등록 화면 이동
        btnVoiceRegister.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, VoiceRegisterActivity.class);
            startActivity(intent);
        });







    }


}