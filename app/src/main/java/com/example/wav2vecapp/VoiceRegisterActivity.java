
package com.example.wav2vecapp;

import android.Manifest;
import android.app.Dialog;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class VoiceRegisterActivity extends AppCompatActivity {

    private Button btnBack, btnStartRecording, btnDeleteRecording;
    private TextView tvKeywordGuide;
    private Spinner spinnerKeywords;

    private SharedPreferences sharedPreferences;
    private String uuid;
    private List<String> keywordList = new ArrayList<>();
    private MediaRecorder recorder;
    private File audioFile;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_voice);

        sharedPreferences = getSharedPreferences("user_info", MODE_PRIVATE);
        uuid = sharedPreferences.getString("uuid", "");
        Log.d("UUID", "📌 UUID 불러오기 결과: " + uuid);

        tvKeywordGuide = findViewById(R.id.tvKeywordGuide);
        btnBack = findViewById(R.id.btnBack);
        btnStartRecording = findViewById(R.id.btnRecord);
        btnDeleteRecording = findViewById(R.id.btnDelete);
        spinnerKeywords = findViewById(R.id.spinnerKeywords);

        loadKeywords(uuid);
        loadKeywordsToSpinner();

        btnBack.setOnClickListener(v -> finish());
        btnStartRecording.setOnClickListener(v -> showRecordStartPopup());
        btnDeleteRecording.setOnClickListener(v -> showRecordDeletePopup());

        // 권한 요청
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1000);
        }
    }

    private void loadKeywords(String uuid) {
        ApiService apiService = RetrofitClient.getApiService();
        KeywordRequest rq = new KeywordRequest(uuid);
        Call<KeywordListResponse> call = apiService.getKeywords(rq);
        call.enqueue(new Callback<KeywordListResponse>() {
            @Override
            public void onResponse(Call<KeywordListResponse> call, Response<KeywordListResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    List<String> keywords = response.body().getKeywords();
                    StringBuilder guide = new StringBuilder("📌 등록된 키워드 목록:\n");
                    for (String keyword : keywords) {
                        guide.append("• ").append(keyword).append("\n");
                    }

                    tvKeywordGuide.setText(guide.toString());
                } else {
                    tvKeywordGuide.setText("❌ 키워드 불러오기 실패");
                }
            }

            @Override
            public void onFailure(Call<KeywordListResponse> call, Throwable t) {
                tvKeywordGuide.setText("❌ 네트워크 오류");
            }
        });
    }

    private void loadKeywordsToSpinner() {
        ApiService apiService = RetrofitClient.getApiService();
        KeywordRequest request = new KeywordRequest(uuid);
        Call<KeywordListResponse> call = apiService.getKeywords(request);
        call.enqueue(new Callback<KeywordListResponse>() {
            @Override
            public void onResponse(Call<KeywordListResponse> call, Response<KeywordListResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    keywordList.clear();
                    keywordList.addAll(response.body().getKeywords());

                    ArrayAdapter<String> adapter = new ArrayAdapter<>(VoiceRegisterActivity.this,
                            android.R.layout.simple_spinner_item, keywordList);
                    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                    spinnerKeywords.setAdapter(adapter);
                    spinnerKeywords.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                        @Override
                        public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                            saveSelectedKeywordToServer();
                        }

                        @Override
                        public void onNothingSelected(AdapterView<?> parent) {}
                    });
                }
            }

            @Override
            public void onFailure(Call<KeywordListResponse> call, Throwable t) {}
        });
    }

    private void saveSelectedKeywordToServer() {
        String selectedKeyword = spinnerKeywords.getSelectedItem().toString();
        SelectedKeywordRequest request = new SelectedKeywordRequest(uuid, selectedKeyword);
        ApiService apiService = RetrofitClient.getApiService();

        Call<ResponseBody> call = apiService.setSelectedKeyword(request);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                Toast.makeText(VoiceRegisterActivity.this, "키워드 저장 완료", Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Toast.makeText(VoiceRegisterActivity.this, "통신 실패", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private boolean isRecording = false;

    private void startRecording() {
        try {
            audioFile = new File(getExternalCacheDir(), "recorded.wav");
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP); // 실제 저장 포맷
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            recorder.setOutputFile(audioFile.getAbsolutePath());
            recorder.prepare();
            recorder.start();
            isRecording = true;
            Log.d("녹음", "녹음 시작됨");
        } catch (Exception e) {
            isRecording = false;
            recorder = null;
            Log.e("녹음 오류", "startRecording 실패: " + e.getMessage());
        }
    }

    private void stopRecordingAndSendToServer() {
        try {
            if (recorder != null && isRecording) {
                recorder.stop();
                recorder.release();
                recorder = null;
                isRecording = false;
                sendToSTTServer(audioFile);
            } else {
                Log.w("녹음 중지", "recorder가 null이거나 녹음 상태가 아님");
            }
        } catch (Exception e) {
            Log.e("녹음 중지 오류", "stopRecording 실패: " + e.getMessage());
        }
    }


    private void sendToSTTServer(File file) {
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(BuildConfig.BACKEND_BASE_URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        ApiService api = retrofit.create(ApiService.class);
        RequestBody requestFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), requestFile);

        Call<ResponseBody> call = api.sendSTT(body);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                if (response.isSuccessful()) {
                    try {
                        String result = response.body().string();
                        Log.d("STT 결과", result);
                        Toast.makeText(VoiceRegisterActivity.this, "전송 완료", Toast.LENGTH_SHORT).show();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Log.e("STT 실패", t.getMessage());
            }
        });
    }

    private void showRecordStartPopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_voice_popup);

        if (dialog.getWindow() != null) {
            dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }

        TextView countText = dialog.findViewById(R.id.tvCount);
        countText.setText("등록 완료 0/4");

        Button btnClose = dialog.findViewById(R.id.btnClose);
        btnClose.setOnClickListener(view -> dialog.dismiss());

        Button btnRecord = dialog.findViewById(R.id.btnRecord);
        Button btnC = dialog.findViewById(R.id.btnC);
        btnRecord.setOnClickListener(view -> startRecording());
        btnC.setOnClickListener(view -> {
            stopRecordingAndSendToServer();
            Toast.makeText(this, "녹음 완료 및 서버 전송", Toast.LENGTH_SHORT).show();
        });

        dialog.show();
    }

    private void showRecordDeletePopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_delete_confirm);

        Button btnYes = dialog.findViewById(R.id.btnYes);
        Button btnNo = dialog.findViewById(R.id.btnNo);
        btnYes.setOnClickListener(v -> dialog.dismiss());
        btnNo.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }
}
