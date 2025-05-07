package com.example.wav2vecapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.File;

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

public class KeywordActivity extends AppCompatActivity {

    private MediaRecorder recorder;
    private String filePath;
    private EditText etKeyword;
    private Button btnAddKeyword, btnStartRecording, btnStopRecording, btnBack;
    private LinearLayout layoutKeywordList;

    private boolean isKeywordRegistering = false;
    private int registerCount = 0;
    private String currentKeyword = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_keyword);

        etKeyword = findViewById(R.id.etKeyword);
        btnAddKeyword = findViewById(R.id.btnAddKeyword);
        btnBack = findViewById(R.id.btnBack);
        btnStartRecording = findViewById(R.id.btnStartRecording);
        btnStopRecording = findViewById(R.id.btnStopRecording);
        layoutKeywordList = findViewById(R.id.layout_keyword_list); // 🔥 키워드 리스트 연결

        filePath = getExternalFilesDir(null).getAbsolutePath() + "/keyword_recorded.wav";

        // 🔙 뒤로가기
        btnBack.setOnClickListener(v -> finish());

        // 🎙️ 녹음 시작 버튼
        btnStartRecording.setOnClickListener(v -> {
            //startKeywordRegistration();
        });

        // ⏹️ 녹음 중지 버튼
        btnStopRecording.setOnClickListener(v -> {
            //stopRecording();
            btnStartRecording.setEnabled(true);
            btnStopRecording.setEnabled(false);
        });

        // ✏️ 키워드 추가 버튼
        btnAddKeyword.setOnClickListener(v -> {
            currentKeyword = etKeyword.getText().toString().trim();
            if (currentKeyword.isEmpty()) {
                Toast.makeText(this, "❗ 키워드를 입력하세요.", Toast.LENGTH_SHORT).show();
                return;
            }
            //addKeywordToList(currentKeyword);   // ✅ 리스트에 추가
            //startKeywordRegistration();         // ✅ 녹음 바로 시작
        });

        // 🔒 권한 요청
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 100);
        }
    }

    // 🎙️ 녹음 시작 함수
    /*private void startRecording() {
        try {
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            recorder.setOutputFile(filePath);

            recorder.prepare();
            recorder.start();

            Toast.makeText(this, "🎙️ 녹음 중...", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "❌ 녹음 실패", Toast.LENGTH_SHORT).show();
        }
    }

    // ⏹️ 녹음 중지 함수
    private void stopRecording() {
        try {
            recorder.stop();
            recorder.release();
            recorder = null;

            Toast.makeText(this, "🎧 녹음 완료", Toast.LENGTH_SHORT).show();

            if (isKeywordRegistering) {
                sendAudioToKeywordRegister(filePath, currentKeyword);
            }

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "❌ 녹음 중지 실패", Toast.LENGTH_SHORT).show();
        }
    }

    // 🚀 서버로 오디오 + 키워드 전송
    private void sendAudioToKeywordRegister(String filePath, String keyword) {
        File file = new File(filePath);
        RequestBody reqFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), reqFile);
        RequestBody keywordBody = RequestBody.create(MediaType.parse("text/plain"), keyword);

        Retrofit retrofit = getRetrofitClient();
        ApiService apiService = retrofit.create(ApiService.class);

        Call<ResponseBody> call = apiService.registerKeyword(body, keywordBody);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                handleKeywordRegistrationResponse(response);
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Toast.makeText(KeywordActivity.this, "🚫 키워드 등록 실패: " + t.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });
    }

    // 🎯 키워드 등록 응답 처리
    private void handleKeywordRegistrationResponse(Response<ResponseBody> response) {
        if (response.isSuccessful()) {
            try {
                String responseBody = response.body().string();
                String msg = new JSONObject(responseBody).getString("message");

                Toast.makeText(this, "✅ " + msg, Toast.LENGTH_SHORT).show();

                registerCount++;
                if (registerCount < 6) {
                    Toast.makeText(this, "🎤 " + (registerCount + 1) + "/6 회차 녹음 시작", Toast.LENGTH_SHORT).show();
                    startRecording();
                    btnStartRecording.setEnabled(false);
                    btnStopRecording.setEnabled(true);
                } else {
                    Toast.makeText(this, "✅ 키워드 6회 등록 완료!", Toast.LENGTH_LONG).show();
                    isKeywordRegistering = false;
                    registerCount = 0;
                }

            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "⚠️ 응답 파싱 오류", Toast.LENGTH_SHORT).show();
            }
        } else {
            Toast.makeText(this, "❌ 서버 오류", Toast.LENGTH_SHORT).show();
        }
    }

    // 🔥 키워드 리스트에 동적으로 추가하는 함수
    private void addKeywordToList(String keyword) {
        LinearLayout newItemLayout = new LinearLayout(this);
        newItemLayout.setOrientation(LinearLayout.HORIZONTAL);
        newItemLayout.setGravity(Gravity.CENTER_VERTICAL);
        newItemLayout.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

        TextView keywordText = new TextView(this);
        keywordText.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                7
        ));
        keywordText.setText(keyword);

        View spacer = new View(this);
        spacer.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                1,
                1
        ));

        CheckBox checkBox = new CheckBox(this);
        checkBox.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                2
        ));
        checkBox.setVisibility(View.GONE);

        newItemLayout.addView(keywordText);
        newItemLayout.addView(spacer);
        newItemLayout.addView(checkBox);

        layoutKeywordList.addView(newItemLayout);
    }*/

    // 🔧 Retrofit 클라이언트 설정
    /*private Retrofit getRetrofitClient() {
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);
        OkHttpClient client = new OkHttpClient.Builder()
                .addInterceptor(logging)
                .build();

        return new Retrofit.Builder()
                .baseUrl(BuildConfig.FLASK_BASE_URL)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
    }

    // 🌟 키워드 등록 시작 함수

    private void startKeywordRegistration() {
        currentKeyword = etKeyword.getText().toString().trim();
        if (currentKeyword.isEmpty()) {
            Toast.makeText(this, "❗ 키워드를 입력하세요.", Toast.LENGTH_SHORT).show();
            return;
        }
        isKeywordRegistering = true;
        registerCount = 0;
        Toast.makeText(this, "🔑 키워드 '" + currentKeyword + "' 등록 시작 (1/6)", Toast.LENGTH_SHORT).show();
        startRecording();
        btnStartRecording.setEnabled(false);
        btnStopRecording.setEnabled(true);
    }*/
}
