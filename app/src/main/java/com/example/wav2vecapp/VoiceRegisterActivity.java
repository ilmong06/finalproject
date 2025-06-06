package com.example.wav2vecapp;

import android.app.Dialog;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class VoiceRegisterActivity extends AppCompatActivity {

    private Button btnBack, btnStartRecording, btnDeleteRecording;
    private TextView tvKeywordGuide;
    private SharedPreferences sharedPreferences;
    private String uuid;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_voice);

        // UUID 가져오기
        sharedPreferences = getSharedPreferences("user_info", MODE_PRIVATE);
        uuid = sharedPreferences.getString("uuid", "");
        Log.d("UUID", "📌 UUID 불러오기 결과: " + uuid);

        // UI 연결
        tvKeywordGuide = findViewById(R.id.tvKeywordGuide);
        btnBack = findViewById(R.id.btnBack);
        btnStartRecording = findViewById(R.id.btnRecord);
        btnDeleteRecording = findViewById(R.id.btnDelete);

        // 키워드 불러오기
        loadKeywords(uuid);

        // 버튼 이벤트
        btnBack.setOnClickListener(v -> finish());
        btnStartRecording.setOnClickListener(v -> showRecordStartPopup());
        btnDeleteRecording.setOnClickListener(v -> showRecordDeletePopup());
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
                    Log.e("Keyword", "서버 오류: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<KeywordListResponse> call, Throwable t) {
                tvKeywordGuide.setText("❌ 네트워크 오류");
                Log.e("Keyword", "API 호출 실패: " + t.getMessage());
            }
        });
    }

    // ======================== 녹음 팝업 관련 ==========================

    private Button btnClose;
    private Button btnRecord, btnC, btnRetry;
    private Button btnFinish;
    private int recordCount = 0;

    private void showRecordStartPopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_voice_popup);

        if (dialog.getWindow() != null) {
            dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }

        TextView countText = dialog.findViewById(R.id.tvCount);
        recordCount = 0;
        countText.setText("등록 완료 0/4");

        btnClose = dialog.findViewById(R.id.btnClose);
        btnClose.setOnClickListener(view -> dialog.dismiss());

        btnRecord = dialog.findViewById(R.id.btnRecord);
        btnRecord.setEnabled(true);
        btnRecord.setAlpha(1.0f);
        btnRecord.setOnClickListener(view -> {
            btnC.setEnabled(true);
            btnC.setAlpha(1.0f);
        });

        btnC = dialog.findViewById(R.id.btnC);
        btnC.setEnabled(false);
        btnC.setAlpha(0.5f);
        btnC.setOnClickListener(view -> {
            if (recordCount < 4) {
                recordCount++;
                countText.setText("등록 완료 " + recordCount + "/4");
            } else {
                Toast.makeText(this, "최대 4개까지 등록 가능합니다.", Toast.LENGTH_SHORT).show();
            }

            btnC.setEnabled(false);
            btnC.setAlpha(0.5f);
        });

        btnRetry = dialog.findViewById(R.id.btnRetry);
        btnRetry.setOnClickListener(view -> {
            recordCount = 0;
            countText.setText("등록 완료 0/4");
        });

        btnFinish = dialog.findViewById(R.id.btnFinish);
        btnFinish.setOnClickListener(view -> dialog.dismiss());

        dialog.setCancelable(true);
        dialog.setCanceledOnTouchOutside(true);
        dialog.show();
    }

    // ======================== 삭제 팝업 관련 ==========================

    private Button btnYes, btnNo;

    private void showRecordDeletePopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_delete_confirm);

        btnYes = dialog.findViewById(R.id.btnYes);
        btnYes.setOnClickListener(v -> dialog.dismiss());

        btnNo = dialog.findViewById(R.id.btnNo);
        btnNo.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }
}
