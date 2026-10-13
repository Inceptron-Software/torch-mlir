// RUN: torch-mlir-opt %s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// RUN: torch-mlir-opt %s -convert-torch-to-linalg -canonicalize -split-input-file -verify-diagnostics > /dev/null

// CHECK-LABEL: func.func @dynamic_rows
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK: tensor.expand_shape
// CHECK: linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<2x?x128xf8E4M3FN>, tensor<2x128x256xf8E4M3FN>)
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.truncf %{{.*}} : f32 to bf16
func.func @dynamic_rows(%a: !torch.vtensor<[?,256],f8E4M3FN>, %b: !torch.vtensor<[256,256],f8E4M3FN>, %sa: !torch.vtensor<[?,2],f32>, %sb: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[?,256],bf16> {
  %a_recipe = torch.constant.int 4
  %b_recipe = torch.constant.int 5
  %dtype = torch.constant.int 15
  %false = torch.constant.bool false
  %none = torch.constant.none
  %as = torch.prim.ListConstruct %sa : (!torch.vtensor<[?,2],f32>) -> !torch.list<vtensor>
  %bs = torch.prim.ListConstruct %sb : (!torch.vtensor<[4,2],f32>) -> !torch.list<vtensor>
  %ra = torch.prim.ListConstruct %a_recipe : (!torch.int) -> !torch.list<int>
  %rb = torch.prim.ListConstruct %b_recipe : (!torch.int) -> !torch.list<int>
  %empty = torch.prim.ListConstruct : () -> !torch.list<int>
  %out = torch.aten._scaled_mm_v2 %a, %b, %as, %ra, %empty, %bs, %rb, %empty, %none, %dtype, %empty, %false : !torch.vtensor<[?,256],f8E4M3FN>, !torch.vtensor<[256,256],f8E4M3FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,256],bf16>
  return %out : !torch.vtensor<[?,256],bf16>
}

// -----

// CHECK-LABEL: func.func @one_row
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK: tensor.expand_shape
// CHECK: linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<1x1x128xf8E4M3FN>, tensor<1x128x128xf8E4M3FN>)
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.truncf %{{.*}} : f32 to bf16
func.func @one_row(%a: !torch.vtensor<[1,128],f8E4M3FN>, %b: !torch.vtensor<[128,128],f8E4M3FN>, %sa: !torch.vtensor<[1,1],f32>, %sb: !torch.vtensor<[4,1],f32>) -> !torch.vtensor<[1,128],bf16> {
  %a_recipe = torch.constant.int 4
  %b_recipe = torch.constant.int 5
  %dtype = torch.constant.int 15
  %false = torch.constant.bool false
  %none = torch.constant.none
  %as = torch.prim.ListConstruct %sa : (!torch.vtensor<[1,1],f32>) -> !torch.list<vtensor>
  %bs = torch.prim.ListConstruct %sb : (!torch.vtensor<[4,1],f32>) -> !torch.list<vtensor>
  %ra = torch.prim.ListConstruct %a_recipe : (!torch.int) -> !torch.list<int>
  %rb = torch.prim.ListConstruct %b_recipe : (!torch.int) -> !torch.list<int>
  %empty = torch.prim.ListConstruct : () -> !torch.list<int>
  %out = torch.aten._scaled_mm_v2 %a, %b, %as, %ra, %empty, %bs, %rb, %empty, %none, %dtype, %empty, %false : !torch.vtensor<[1,128],f8E4M3FN>, !torch.vtensor<[128,128],f8E4M3FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,128],bf16>
  return %out : !torch.vtensor<[1,128],bf16>
}

// -----

// CHECK-LABEL: func.func @padded_k_groups
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK: tensor.expand_shape
// CHECK: linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<5x33x128xf8E4M3FN>, tensor<5x128x128xf8E4M3FN>)
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.truncf %{{.*}} : f32 to bf16
func.func @padded_k_groups(%a: !torch.vtensor<[33,640],f8E4M3FN>, %b: !torch.vtensor<[640,128],f8E4M3FN>, %sa: !torch.vtensor<[33,5],f32>, %sb: !torch.vtensor<[8,1],f32>) -> !torch.vtensor<[33,128],bf16> {
  %a_recipe = torch.constant.int 4
  %b_recipe = torch.constant.int 5
  %dtype = torch.constant.int 15
  %false = torch.constant.bool false
  %none = torch.constant.none
  %as = torch.prim.ListConstruct %sa : (!torch.vtensor<[33,5],f32>) -> !torch.list<vtensor>
  %bs = torch.prim.ListConstruct %sb : (!torch.vtensor<[8,1],f32>) -> !torch.list<vtensor>
  %ra = torch.prim.ListConstruct %a_recipe : (!torch.int) -> !torch.list<int>
  %rb = torch.prim.ListConstruct %b_recipe : (!torch.int) -> !torch.list<int>
  %empty = torch.prim.ListConstruct : () -> !torch.list<int>
  %out = torch.aten._scaled_mm_v2 %a, %b, %as, %ra, %empty, %bs, %rb, %empty, %none, %dtype, %empty, %false : !torch.vtensor<[33,640],f8E4M3FN>, !torch.vtensor<[640,128],f8E4M3FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[33,128],bf16>
  return %out : !torch.vtensor<[33,128],bf16>
}

// -----

func.func @bad_scale_padding(%a: !torch.vtensor<[2,256],f8E4M3FN>, %b: !torch.vtensor<[256,128],f8E4M3FN>, %sa: !torch.vtensor<[2,2],f32>, %sb: !torch.vtensor<[2,1],f32>) -> !torch.vtensor<[2,128],bf16> {
  %a_recipe = torch.constant.int 4
  %b_recipe = torch.constant.int 5
  %dtype = torch.constant.int 15
  %false = torch.constant.bool false
  %none = torch.constant.none
  %as = torch.prim.ListConstruct %sa : (!torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  %bs = torch.prim.ListConstruct %sb : (!torch.vtensor<[2,1],f32>) -> !torch.list<vtensor>
  %ra = torch.prim.ListConstruct %a_recipe : (!torch.int) -> !torch.list<int>
  %rb = torch.prim.ListConstruct %b_recipe : (!torch.int) -> !torch.list<int>
  %empty = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten._scaled_mm_v2'}}
  %out = torch.aten._scaled_mm_v2 %a, %b, %as, %ra, %empty, %bs, %rb, %empty, %none, %dtype, %empty, %false : !torch.vtensor<[2,256],f8E4M3FN>, !torch.vtensor<[256,128],f8E4M3FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,128],bf16>
  return %out : !torch.vtensor<[2,128],bf16>
}

// -----

func.func @bad_scale_rows(%a: !torch.vtensor<[2,256],f8E4M3FN>, %b: !torch.vtensor<[256,128],f8E4M3FN>, %sa: !torch.vtensor<[3,2],f32>, %sb: !torch.vtensor<[4,1],f32>) -> !torch.vtensor<[2,128],bf16> {
  %a_recipe = torch.constant.int 4
  %b_recipe = torch.constant.int 5
  %dtype = torch.constant.int 15
  %false = torch.constant.bool false
  %none = torch.constant.none
  %as = torch.prim.ListConstruct %sa : (!torch.vtensor<[3,2],f32>) -> !torch.list<vtensor>
  %bs = torch.prim.ListConstruct %sb : (!torch.vtensor<[4,1],f32>) -> !torch.list<vtensor>
  %ra = torch.prim.ListConstruct %a_recipe : (!torch.int) -> !torch.list<int>
  %rb = torch.prim.ListConstruct %b_recipe : (!torch.int) -> !torch.list<int>
  %empty = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten._scaled_mm_v2'}}
  %out = torch.aten._scaled_mm_v2 %a, %b, %as, %ra, %empty, %bs, %rb, %empty, %none, %dtype, %empty, %false : !torch.vtensor<[2,256],f8E4M3FN>, !torch.vtensor<[256,128],f8E4M3FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,128],bf16>
  return %out : !torch.vtensor<[2,128],bf16>
}

// -----

func.func @unsupported_recipe_pair(%a: !torch.vtensor<[2,256],f8E4M3FN>, %b: !torch.vtensor<[256,128],f8E4M3FN>, %sa: !torch.vtensor<[2,2],f32>, %sb: !torch.vtensor<[4,1],f32>) -> !torch.vtensor<[2,128],bf16> {
  %a_recipe = torch.constant.int 5
  %b_recipe = torch.constant.int 4
  %dtype = torch.constant.int 15
  %false = torch.constant.bool false
  %none = torch.constant.none
  %as = torch.prim.ListConstruct %sa : (!torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  %bs = torch.prim.ListConstruct %sb : (!torch.vtensor<[4,1],f32>) -> !torch.list<vtensor>
  %ra = torch.prim.ListConstruct %a_recipe : (!torch.int) -> !torch.list<int>
  %rb = torch.prim.ListConstruct %b_recipe : (!torch.int) -> !torch.list<int>
  %empty = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten._scaled_mm_v2'}}
  %out = torch.aten._scaled_mm_v2 %a, %b, %as, %ra, %empty, %bs, %rb, %empty, %none, %dtype, %empty, %false : !torch.vtensor<[2,256],f8E4M3FN>, !torch.vtensor<[256,128],f8E4M3FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,128],bf16>
  return %out : !torch.vtensor<[2,128],bf16>
}
