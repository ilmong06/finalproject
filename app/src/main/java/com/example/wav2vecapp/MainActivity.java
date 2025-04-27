package com.example.wav2vecapp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.File;
import java.util.List;

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

public class MainActivity extends AppCompatActivity {

    private MediaRecorder recorder;
    private String filePath;
    private TextView textView, textRegisterStep;
    private EditText keywordInput;
    private Button startButton, stopButton, registerButton, keywordRegisterButton;

    private boolean isRegistering = false;
    private boolean isKeywordRegistering = false;
    private int registerCount = 0;
    private String currentKeyword = "";
    private LocationHelper locationHelper;
    private Button locationButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button btnMoveKeywordPage = findViewById(R.id.btnMoveKeywordPage);
        btnMoveKeywordPage.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, KeywordActivity.class);
            startActivity(intent);
        });

        textView = findViewById(R.id.textResult);
        textRegisterStep = findViewById(R.id.textRegisterStep);
        keywordInput = findViewById(R.id.editKeyword);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);
        registerButton = findViewById(R.id.registerButton);
        keywordRegisterButton = findViewById(R.id.keywordRegisterButton);
        locationHelper = new LocationHelper(this, textView, textRegisterStep);
// onCreate 내부에서 버튼 연결
        locationButton = findViewById(R.id.locationButton);
        locationHelper = new LocationHelper(this, textView, textRegisterStep);

        locationButton.setOnClickListener(view -> {
            Log.i("MainActivity", "🟡 위치 버튼 클릭됨");
            locationHelper.requestLocationPermission();
        });

        filePath = getExternalFilesDir(null).getAbsolutePath() + "/recorded.wav";

        startButton.setOnClickListener(view -> {
            isRegistering = false;
            isKeywordRegistering = false;
            startRecording();
        });

        stopButton.setOnClickListener(view -> stopRecording());

        registerButton.setOnClickListener(view -> {
            isRegistering = true;
            isKeywordRegistering = false;
            registerCount = 0;
            textRegisterStep.setText("🧬 화자 1/4 회차 등록 시작");
            startRecording();
        });

        keywordRegisterButton.setOnClickListener(view -> {
            currentKeyword = keywordInput.getText().toString().trim();
            if (currentKeyword.isEmpty()) {
                textView.setText("❗ 먼저 키워드를 입력하세요.");
                return;
            }
            isRegistering = false;
            isKeywordRegistering = true;
            registerCount = 0;
            textRegisterStep.setText("🔑 키워드 '" + currentKeyword + "' 1/6 등록 시작");
            startRecording();
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.ACCESS_FINE_LOCATION
            }, 200);
        }

    }

    private void startRecording() {
        try {
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            recorder.setOutputFile(filePath);

            recorder.prepare();
            recorder.start();

            textView.setText("🎙️ 녹음 중... (Tap 종료)");
            stopButton.setEnabled(true);
            startButton.setEnabled(false);
            registerButton.setEnabled(false);
            keywordRegisterButton.setEnabled(false);

        } catch (Exception e) {
            e.printStackTrace();
            textView.setText("❌ 녹음 시작 실패");
        }
    }

    private void stopRecording() {
        try {
            recorder.stop();
            recorder.release();
            recorder = null;

            textView.setText("🎧 녹음 종료됨");

            if (isRegistering) {
                sendAudioToRegister(filePath);
            } else if (isKeywordRegistering) {
                sendAudioToKeywordRegister(filePath, currentKeyword);
            } else {
                sendAudioToServer(filePath);
            }

            stopButton.setEnabled(false);
            startButton.setEnabled(true);
            registerButton.setEnabled(true);
            keywordRegisterButton.setEnabled(true);

        } catch (Exception e) {
            e.printStackTrace();
            textView.setText("❌ 녹음 중지 실패");
        }
    }

    private void sendAudioToServer(String filePath) {
        File file = new File(filePath);
        RequestBody reqFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), reqFile);

        Retrofit retrofit = getRetrofitClient();
        ApiService apiService = retrofit.create(ApiService.class);
        Call<TranscriptionResponse> call = apiService.uploadAudio(body);

        call.enqueue(new Callback<TranscriptionResponse>() {
            @Override
            public void onResponse(Call<TranscriptionResponse> call, Response<TranscriptionResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("📝 텍스트 결과:\n").append(response.body().text).append("\n\n");

                    if (response.body().triggeredKeyword != null) {
                        sb.append("🔑 키워드: ").append(response.body().triggeredKeyword).append("\n\n");
                    }

                    List<Float> speakerVector = response.body().speakerVector;
                    if (speakerVector != null) {
                        sb.append("🧬 화자 벡터:\n");
                        for (Float val : speakerVector) {
                            sb.append(String.format("%.4f ", val));
                        }
                    }
                    textView.setText(sb.toString());
                } else {
                    textView.setText("❌ 서버 오류 또는 인증 실패");
                }
            }

            @Override
            public void onFailure(Call<TranscriptionResponse> call, Throwable t) {
                textView.setText("🚫 연결 실패: " + t.getMessage());
            }
        });
    }

    private void sendAudioToRegister(String filePath) {
        File file = new File(filePath);
        RequestBody reqFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), reqFile);

        Retrofit retrofit = getRetrofitClient();
        ApiService apiService = retrofit.create(ApiService.class);

        Call<ResponseBody> call = apiService.registerSpeaker(body);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                handleRegistrationResponse(response, "화자", 4);
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                textView.setText("🚫 등록 요청 실패: " + t.getMessage());
            }
        });
    }

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
                handleRegistrationResponse(response, "키워드", 6);
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                textView.setText("🚫 키워드 등록 실패: " + t.getMessage());
            }
        });
    }

    private void handleRegistrationResponse(Response<ResponseBody> response, String type, int maxCount) {
        if (response.isSuccessful()) {
            try {
                String responseBody = response.body().string();
                String msg = new JSONObject(responseBody).getString("message");

                textView.setText("✅ " + msg);
                textRegisterStep.setText(msg);

                if (msg.contains(maxCount + "/" + maxCount)) {
                    isRegistering = false;
                    isKeywordRegistering = false;
                    registerCount = 0;
                } else {
                    registerCount++;
                    if (registerCount < maxCount) {
                        textView.setText("🎤 " + (registerCount + 1) + "/" + maxCount + " 회차 녹음 시작");
                        startRecording();
                    }
                }

            } catch (Exception e) {
                textView.setText("⚠️ 응답 파싱 오류");
                e.printStackTrace();
            }
        } else {
            textView.setText("❌ 등록 실패 (서버 오류)");
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        locationHelper.onRequestPermissionsResult(requestCode, grantResults);
    }


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
}
