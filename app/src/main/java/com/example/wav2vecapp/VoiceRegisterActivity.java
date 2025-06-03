package com.example.wav2vecapp;

import android.app.Dialog;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.view.Window;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.atomic.AtomicInteger;

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

    /*
     * 본 메소드는 음성 등록 팝업창의 버튼이벤트입니다.
     * */
    /**
     * 위에서 차례대로
     * 닫기
     * 녹음, 중지, 초기화
     * 완료
     */
    private Button btnClose;
    private Button btnRecord, btnC, btnRetry;
    private Button btnFinish;


    /// 음성등록 팝업창의 기능들
    ///
    private int recordCount = 0; // 등록된 음성 수 카운트 (0~4)

    private void showRecordStartPopup() {


        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_voice_popup);

        if (dialog.getWindow() != null) {
            dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }

        // ✅ TextView 및 초기 텍스트 설정
        TextView countText = dialog.findViewById(R.id.tvCount);
        recordCount = 0; // 팝업 열릴 때 초기화
        countText.setText("등록 완료 0/4");

        // ✅ 닫기 버튼
        btnClose = dialog.findViewById(R.id.btnClose);
        btnClose.setOnClickListener(view -> dialog.dismiss());

        if(recordCount == 4){
            btnRecord.setEnabled(false);
            btnRecord.setAlpha(0.5f);
            btnC.setEnabled(false);
            btnC.setAlpha(0.5f);
        }else{
            btnRecord.setEnabled(true);
            btnRecord.setAlpha(1.0f);
            btnC.setEnabled(true);
            btnC.setAlpha(1.0f);
        }

        // ✅ 녹음 시작 버튼
        btnRecord = dialog.findViewById(R.id.btnRecord);
        btnRecord.setOnClickListener(view -> {
            // 음성 녹음 시작


            btnC.setEnabled(true);
            btnC.setAlpha(1.0f);
        });

        // ✅ 녹음 중지 버튼
        btnC = dialog.findViewById(R.id.btnC);
        btnC.setOnClickListener(view -> {
            if (recordCount < 4) {
                recordCount++;
                countText.setText("등록 완료 " + recordCount + "/4");
            } else {
                Toast.makeText(this, "최대 4개까지 등록 가능합니다.", Toast.LENGTH_SHORT).show();
            }

            btnC.setEnabled(false);
            btnC.setAlpha(0.5f);

            // 음성 데이터 축적 처리
        });

        // ✅ 녹음 초기화 버튼
        btnRetry = dialog.findViewById(R.id.btnRetry);
        btnRetry.setOnClickListener(view -> {
            recordCount = 0;
            countText.setText("등록 완료 0/4");

            // 음성 데이터 삭제 및 초기화 처리
        });

        // ✅ 등록 완료 버튼
        btnFinish = dialog.findViewById(R.id.btnFinish);
        btnFinish.setOnClickListener(view -> {
            // 음성 데이터 저장 및 서버 전송 등 처리

            dialog.dismiss();
        });

        dialog.setCancelable(true);
        dialog.setCanceledOnTouchOutside(true);
        dialog.show();
    }


    /**
     * 🗑️ 녹음 삭제 확인 팝업 표시
     * - 팝업 레이아웃: activity_delete_confirm.xml
     */


    private Button btnYes, btnNo;

    private void showRecordDeletePopup() {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.activity_delete_confirm);  // 📄 사용자 정의 팝업 레이아웃

        btnYes = dialog.findViewById(R.id.btnYes);
        btnYes.setOnClickListener(v -> {
        //     // 🔥 여기에 실제 삭제 로직 추가
           dialog.dismiss();
        });

        btnNo = dialog.findViewById(R.id.btnNo);
        btnNo.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }
}


