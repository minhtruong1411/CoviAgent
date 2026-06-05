"""
Làm sạch đồ thị: GỘP các node __Entity__ trùng nghĩa (entity resolution).

Đồ thị được trích tự động nên có nhiều node trùng (số nhiều/ít, biến thể tên),
khiến facts bị lặp và quan hệ bị phân mảnh trên nhiều node → GraphRAG kém completeness.
Script này gộp các biến thể về một node chính (canonical), chuyển hướng toàn bộ quan
hệ (HAS_ENTITY, CAUSES, TREATS, ...) về node đó bằng apoc.refactor.mergeNodes.

AN TOÀN:
- Chỉ gộp các nhóm đã được kiểm tra thủ công bên dưới (MERGE_GROUPS).
- KHÔNG gộp Covid-19 (bệnh) với Sars-Cov-2 (virus) — đó là 2 khái niệm khác nhau.
- Idempotent: chạy lại nhiều lần vẫn an toàn (node đã gộp thì bỏ qua).
- Cần APOC (đã kiểm tra có sẵn).

Usage:
  python scripts/clean_graph_entities.py            # gộp thật
  python scripts/clean_graph_entities.py --dry-run  # chỉ in kế hoạch, không sửa
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Mỗi nhóm: phần tử ĐẦU là node GIỮ LẠI (canonical), các phần tử sau bị gộp vào.
# Canonical chọn theo: bậc liên kết cao nhất + tên rõ nghĩa nhất.
MERGE_GROUPS = [
    ["Covid-19 Test", "Sars-Cov-2 Test", "Test"],
    ["Antigen Tests", "Antigen Test"],
    ["Covid-19 Vaccine", "Covid-19 Vaccines"],
    ["Infection", "Sars-Cov-2 Infection"],
    ["Naat", "Naats"],
    ["Virus", "Sars-Cov-2 Virus"],
    ["Severe Illness", "Severe Covid-19 Illness"],
    ["Triệu Chứng", "Triệu Chứng Covid-19"],
    # Pass 2 — cụm xét nghiệm bị phân mảnh nặng (gồm trùng SONG NGỮ Việt/Anh).
    # GIỮ NGUYÊN các LOẠI test khác nhau: NAAT, PCR, Antigen, Viral Tests là riêng biệt.
    ["Naat", "Naats Tests", "Nucleic Acid Amplification Tests (Naats)"],
    ["Pcr Tests", "Pcr"],
    ["Testing", "Testing For Covid-19"],
    ["Covid-19 Test", "Xét Nghiệm Covid-19"],
    ["Antigen Tests", "Xét Nghiệm Kháng Nguyên"],

    # ── Pass 3: gộp SONG NGỮ (Việt/Anh) + paraphrase. Đồ thị này song ngữ nên
    #    cùng khái niệm bị tách thành node Việt và node Anh → phân mảnh nặng.
    #    NGUYÊN TẮC: vẫn GIỮ PHÂN BIỆT từng yếu tố nguy cơ, từng loại thuốc, từng loại test.

    # Mức độ NẶNG — quan trọng nhất: gom mọi yếu tố nguy cơ về CÙNG một node 'Severe Illness'
    ["Severe Illness", "Severe Covid-19", "Serious Illness", "Serious", "Bệnh Nặng", "Nghiêm Trọng", "Getting Very Sick"],
    # Mức độ NHẸ/trung bình
    ["Mild To Moderate", "Mild", "Mild Illness", "Moderate", "Nhẹ", "Nhẹ Đến Trung Bình"],
    ["Death", "Dying", "Tử Vong"],
    ["Hospitalization", "Hospitalized"],

    # Yếu tố nguy cơ — gộp song ngữ, GIỮ phân biệt từng yếu tố
    ["Older Adults", "Older People", "Người Lớn Tuổi"],
    ["Underlying Medical Conditions", "Bệnh Lý Nền"],
    ["Diabetes", "Tiểu Đường"],
    ["Cancer", "Ung Thư"],
    ["Heart Disease", "Cardiovascular Disease", "Bệnh Tim Mạch"],
    ["Chronic Lung Disease", "Chronic Respiratory Disease", "Bệnh Hô Hấp Mãn Tính", "Bệnh Lý Hô Hấp"],

    # Biện pháp phòng ngừa — gộp paraphrase + song ngữ, GIỮ phân biệt từng biện pháp
    ["Physical Distancing", "Social Distancing", "Social Distancing (1 Meter)", "Staying 1 Meter Apart", "Putting Distance Between Yourself And Others", "Keeping Distance From Others", "Giữ Khoảng Cách"],
    ["Wearing A Mask", "Đeo Khẩu Trang"],
    ["Hand Hygiene", "Hand Washing/Alcohol Rub", "Cleaning Hands With Alcohol-Based Hand Rub", "Washing Hands With Soap And Water"],
    ["Respiratory Etiquette", "Covering Mouth And Nose", "Covering Mouth And Nose When Coughing Or Sneezing"],
    ["Staying Home", "Self-Isolation", "Staying Home And Self-Isolating"],
    ["Ventilation", "Cleaner Air"],
    ["Vaccination", "Covid-19 Vaccine"],
    ["Phòng Ngừa", "Chiến Lược Phòng Ngừa", "Preventive Measures", "Giảm Nguy Cơ Lây Lan"],

    # Điều trị / chăm sóc y tế
    ["Treatment", "Special Treatment", "Điều Trị"],
    ["Medical Attention", "Emergency Care", "Seeking Immediate Medical Attention", "Chăm Sóc Y Tế"],
    ["Managing Symptoms At Home", "Kiểm Soát Triệu Chứng Tại Nhà"],
    ["Paxlovid", "Nirmatrelvir With Ritonavir (Paxlovid)"],

    # ── Pass 4: triệu chứng trùng song ngữ (giữ phân biệt từng triệu chứng)
    ["Cough", "Ho"],
    ["Chest Pain", "Đau Ngực"],
    ["Difficulty Breathing", "Shortness Of Breath", "Khó Thở Hoặc Hụt Hơi"],
    ["Loss Of Speech Or Mobility", "Mất Khả Năng Nói Hoặc Vận Động Hoặc Nhầm Lẫn"],
    ["Triệu Chứng", "Symptoms"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch, không sửa graph")
    args = ap.parse_args()

    load_dotenv()
    g = Neo4jGraph(
        url=os.environ.get("NEO4J_URI"),
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD"),
        database=os.environ.get("NEO4J_DATABASE"),
    )

    before = g.query("MATCH (e:__Entity__) RETURN count(e) AS n")[0]["n"]
    print(f"Tổng __Entity__ trước: {before}")
    print("=" * 64)

    merged_total = 0
    for group in MERGE_GROUPS:
        keep, dups = group[0], group[1:]
        # chỉ giữ những dup THỰC SỰ còn tồn tại (idempotent)
        existing = g.query(
            "MATCH (d:__Entity__) WHERE d.id IN $dups RETURN collect(d.id) AS ids",
            params={"dups": dups},
        )[0]["ids"]
        keep_exists = g.query(
            "MATCH (k:__Entity__ {id:$id}) RETURN count(k) AS n", params={"id": keep}
        )[0]["n"]

        if not keep_exists:
            print(f"⏭️  Bỏ qua (không thấy canonical): {keep}")
            continue
        if not existing:
            print(f"✅ Đã sạch: {keep}  (không còn biến thể trùng)")
            continue

        print(f"🔗 Gộp vào '{keep}'  <-  {existing}")
        merged_total += len(existing)
        if args.dry_run:
            continue

        # mergeNodes: node đầu giữ nguyên id/embedding (properties:'discard'),
        # mergeRels:true → gộp các quan hệ trùng (cùng type+endpoint) làm một.
        g.query(
            """
            MATCH (keep:__Entity__ {id:$keep})
            MATCH (d:__Entity__) WHERE d.id IN $dups
            WITH keep, collect(d) AS ds
            CALL apoc.refactor.mergeNodes([keep] + ds,
                 {properties:'discard', mergeRels:true}) YIELD node
            RETURN node.id AS id
            """,
            params={"keep": keep, "dups": existing},
        )

    print("=" * 64)
    after = g.query("MATCH (e:__Entity__) RETURN count(e) AS n")[0]["n"]
    action = "SẼ gộp" if args.dry_run else "Đã gộp"
    print(f"{action} {merged_total} node trùng.")
    if not args.dry_run:
        print(f"Tổng __Entity__ sau: {after}  (giảm {before - after})")
        # kiểm chứng nhanh: Antigen Tests giờ gom đủ cạnh DETECTS chưa
        chk = g.query(
            "MATCH (e:__Entity__ {id:'Antigen Tests'})-[r]-() RETURN count(r) AS deg"
        )
        if chk:
            print(f"Kiểm chứng: 'Antigen Tests' giờ có {chk[0]['deg']} quan hệ (gồm cả của biến thể).")


if __name__ == "__main__":
    main()
