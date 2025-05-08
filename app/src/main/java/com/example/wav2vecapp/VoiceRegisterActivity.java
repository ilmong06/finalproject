package com.example.wav2vecapp;

import android.app.Dialog;
import android.os.Bundle;
import android.view.Window;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

/**
 * 🔊 음성 등록 화면 액티비티
 * - 음성 녹음 시작 및 삭제 관련 팝업 UI 처리
 */
public class VoiceRegisterActivity extends AppCompatActivity {

    private Button btnBack, btnStartRecording, btnDeleteRecording;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_voice);

        // 🔗 버튼 연결
        btnBack = findViewById(R.id.btnBack);
        btnStartRecording = findViewById(R.id.btnRecord);
        btnDeleteRecording = findViewById(R.id.btnDelete);

        // 🔙 1) 뒤로가기 버튼 → 현재 화면 종료
        btnBack.setOnClickListener(v -> finish());

        // 🎙️ 2) 음성 녹음 시작 → 커스텀 팝업창 표시
        btnStartRecording.setOnClickListener(v -> showRecordStartPopup());

        // 🗑️ 3) 음성 삭제 → 삭제 확인 팝업 표시
        btnDeleteRecording.setOnClickListener(v -> showRecordDeletePopup());
    }

    /**
     * 🎤 녹음 시작 안내 팝업 표시
     * - 팝업 레이아웃: activity_voice_popup.xml
     */
    private void showRecordStartPopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_voice_popup);  // 📄 사용자 정의 팝업 레이아웃

        // 팝업 닫기 버튼이 있다면 여기서 dismiss 처리 추가 가능
        // Button btnClose = dialog.findViewById(R.id.btnClose);
        // btnClose.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }

    /**
     * 🗑️ 녹음 삭제 확인 팝업 표시
     * - 팝업 레이아웃: activity_delete_confirm.xml
     */
    private void showRecordDeletePopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_delete_confirm);  // 📄 사용자 정의 팝업 레이아웃

        // Button btnDeleteOk = dialog.findViewById(R.id.btnConfirm);
        // btnDeleteOk.setOnClickListener(v -> {
        //     // 🔥 여기에 실제 삭제 로직 추가
        //     dialog.dismiss();
        // });

        dialog.show();
    }
}


