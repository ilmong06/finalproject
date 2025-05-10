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

    // 🎙️ 녹음 관련 변수
    private MediaRecorder recorder;
    private String filePath;

    // 🔧 UI 컴포넌트
    private EditText etKeyword;
    private Button btnAddKeyword, btnStartRecording, btnStopRecording, btnBack;
    private LinearLayout layoutKeywordList;

    // 🔁 녹음 상태 변수
    private boolean isKeywordRegistering = false;
    private int registerCount = 0;
    private String currentKeyword = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_keyword);

        // 🔗 UI 요소 연결
        etKeyword = findViewById(R.id.etKeyword);
        btnAddKeyword = findViewById(R.id.btnAddKeyword);
        btnBack = findViewById(R.id.btnBack);
        btnStartRecording = findViewById(R.id.btnStartRecording);
        btnStopRecording = findViewById(R.id.btnStopRecording);
        layoutKeywordList = findViewById(R.id.layout_keyword_list);

        // 🎙️ 녹음 파일 저장 경로 설정
        filePath = getExternalFilesDir(null).getAbsolutePath() + "/keyword_recorded.wav";

        // 🔙 뒤로가기 버튼
        btnBack.setOnClickListener(v -> finish());

        // ▶️ 키워드 녹음 시작 버튼
        btnStartRecording.setOnClickListener(v -> startKeywordRegistration());

        // ⏹️ 키워드 녹음 중지 버튼
        btnStopRecording.setOnClickListener(v -> {
            stopRecording();
            btnStartRecording.setEnabled(true);
            btnStopRecording.setEnabled(false);
        });

        // ➕ 키워드 추가 및 녹음 시작
        btnAddKeyword.setOnClickListener(v -> {
            currentKeyword = etKeyword.getText().toString().trim();
            if (currentKeyword.isEmpty()) {
                Toast.makeText(this, "❗ 키워드를 입력하세요.", Toast.LENGTH_SHORT).show();
                return;
            }
            addKeywordToList(currentKeyword); // 동적 리스트 추가
            startKeywordRegistration();       // 바로 녹음 시작
        });

        // 🔐 마이크 및 저장소 권한 요청
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 100);
        }
    }

    // 🎙️ 실제 녹음 시작 처리
    private void startRecording() {
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

    // ⏹️ 녹음 종료 및 서버 전송
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

    // 🚀 서버에 오디오 파일 + 키워드 텍스트 전송
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

    // ✅ 서버 응답 처리 (6회 완료 시 종료)
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

    // 🔧 Retrofit 설정
    private Retrofit getRetrofitClient() {
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

    // 🟡 키워드 등록 시작 흐름
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
    }

    // ✅ 키워드 리스트 UI에 추가
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
        checkBox.setVisibility(View.GONE); // 체크박스는 현재 안보이도록 설정

        newItemLayout.addView(keywordText);
        newItemLayout.addView(spacer);
        newItemLayout.addView(checkBox);

        layoutKeywordList.addView(newItemLayout);
    }
}
