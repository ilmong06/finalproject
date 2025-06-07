
package com.example.wav2vecapp;

import android.Manifest;
import android.app.Dialog;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.media.AudioFormat;
import android.media.AudioRecord;
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
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
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
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private File wavFile;
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


    private void startRecording() {
        int sampleRate = 16000;
        int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
        int bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return;
        }
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,
                sampleRate, channelConfig, audioFormat, bufferSize);

        wavFile = new File(getExternalCacheDir(), "recorded.wav");

        isRecording = true;
        audioRecord.startRecording();

        new Thread(() -> {
            try (FileOutputStream os = new FileOutputStream(wavFile)) {
                byte[] buffer = new byte[bufferSize];
                while (isRecording) {
                    int read = audioRecord.read(buffer, 0, buffer.length);
                    if (read > 0) {
                        os.write(buffer, 0, read);
                    }
                }
            } catch (IOException e) {
                Log.e("녹음 오류", "파일 저장 실패: " + e.getMessage());
            }
        }).start();

        Log.d("녹음", "WAV 녹음 시작");
    }

    private void stopRecordingAndSendToServer() {
        if (audioRecord != null && isRecording) {
            isRecording = false;
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;

            // WAV 헤더 붙이기
            try {
                File wavWithHeader = new File(getExternalCacheDir(), "final_recorded.wav");
                addWavHeader(wavFile, wavWithHeader, 16000, 1, 16);
                sendToSTTServer(wavWithHeader);
            } catch (IOException e) {
                Log.e("WAV 변환 오류", e.getMessage());
            }
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
    private void addWavHeader(File pcmFile, File wavFile, int sampleRate, int channels, int bitsPerSample) throws IOException {
        byte[] pcmData = null;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            pcmData = Files.readAllBytes(pcmFile.toPath());
        }
        int byteRate = sampleRate * channels * bitsPerSample / 8;

        try (FileOutputStream out = new FileOutputStream(wavFile)) {
            out.write("RIFF".getBytes());
            out.write(intToLittleEndian(36 + pcmData.length));
            out.write("WAVEfmt ".getBytes());
            out.write(intToLittleEndian(16)); // Subchunk1Size
            out.write(shortToLittleEndian((short) 1)); // PCM
            out.write(shortToLittleEndian((short) channels));
            out.write(intToLittleEndian(sampleRate));
            out.write(intToLittleEndian(byteRate));
            out.write(shortToLittleEndian((short) (channels * bitsPerSample / 8))); // Block align
            out.write(shortToLittleEndian((short) bitsPerSample));
            out.write("data".getBytes());
            out.write(intToLittleEndian(pcmData.length));
            out.write(pcmData);
        }
    }

    private byte[] intToLittleEndian(int value) {
        return new byte[]{(byte) value, (byte) (value >> 8), (byte) (value >> 16), (byte) (value >> 24)};
    }

    private byte[] shortToLittleEndian(short value) {
        return new byte[]{(byte) value, (byte) (value >> 8)};
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
