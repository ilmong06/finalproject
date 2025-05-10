package com.example.wav2vecapp;

import android.Manifest;
import android.app.Dialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.File;
import java.util.List;
import java.util.Objects;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.InputFilter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

/*
* 본 클래스는 음성 등록 팝업창의 버튼이벤트 클래스입니다.
* */
public class VoiceRegister extends AppCompatActivity{

    // MainActivity.java 또는 RegisterVoiceActivity.java
    Button popupTrigger;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_voice_popup);

        popupTrigger = findViewById(R.id.btnRecord);// 메인 화면의 녹음 버튼

        popupTrigger.setOnClickListener(v -> {
            Dialog dialog = new Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen);
            dialog.setContentView(R.layout.activity_voice_popup);
            Objects.requireNonNull(dialog.getWindow()).setBackgroundDrawableResource(android.R.color.transparent);
            dialog.show();

            // 팝업 내부 버튼 이벤트 처리
            ImageButton btnClose = dialog.findViewById(R.id.btnClose);
            btnClose.setOnClickListener(view -> dialog.dismiss());

            Button btnRecord = dialog.findViewById(R.id.btnRecord);
            btnRecord.setOnClickListener(view -> {
                // 녹음 시작 처리
            });

            Button btnC = dialog.findViewById(R.id.btnC);
            Button btnRetry = dialog.findViewById(R.id.btnRetry);
            Button btnFinish = dialog.findViewById(R.id.btnFinish);
            btnFinish.setOnClickListener(view -> {
                // 등록 완료 처리
                dialog.dismiss();
            });
        });
    }




}
