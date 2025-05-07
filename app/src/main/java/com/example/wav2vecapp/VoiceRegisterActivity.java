package com.example.wav2vecapp;

import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.view.Window;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

/*
* 본 클래스는 음성 등록 화면의 이벤트 클래스입니다.*/

public class VoiceRegisterActivity extends AppCompatActivity {

    private Button btnBack, btnStartRecording, btnDeleteRecording;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_voice);

        btnBack = findViewById(R.id.btnBack);
        btnStartRecording = findViewById(R.id.btnRecord);
        btnDeleteRecording = findViewById(R.id.btnDelete);

        // 1) 뒤로가기 버튼
        btnBack.setOnClickListener(v -> {
            finish(); // 현재 액티비티 종료 → 이전 화면(MainActivity)로 돌아감
        });

        // 2) 음성 시작 버튼
        btnStartRecording.setOnClickListener(v -> showRecordStartPopup());

        // 3) 음성 삭제 버튼
        btnDeleteRecording.setOnClickListener(v -> showRecordDeletePopup());
    }

    private void showRecordStartPopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_voice_popup);  // 직접 만든 팝업 레이아웃


        dialog.show();
    }

    private void showRecordDeletePopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_delete_confirm);  // 삭제 확인용 팝업 레이아웃

        dialog.show();
    }
}

