import AVFoundation
import AudioToolbox

let args = CommandLine.arguments
guard args.count >= 4 else {
    FileHandle.standardError.write("usage: decode_track <in> <out.raw> <audioTrackIndex>\n".data(using: .utf8)!)
    exit(2)
}
let src = URL(fileURLWithPath: args[1])
let dstPath = args[2]
let want = Int(args[3])!

let asset = AVURLAsset(url: src)
let audioTracks = asset.tracks(withMediaType: .audio)
guard want < audioTracks.count else {
    FileHandle.standardError.write("only \(audioTracks.count) audio tracks\n".data(using: .utf8)!)
    exit(3)
}
let track = audioTracks[want]

var channels: UInt32 = 0
var sampleRate: Float64 = 0
var layoutData: Data? = nil
if let fds = track.formatDescriptions as? [CMFormatDescription], let fd = fds.first {
    if let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fd)?.pointee {
        channels = asbd.mChannelsPerFrame
        sampleRate = asbd.mSampleRate
    }
    var size = 0
    if let acl = CMAudioFormatDescriptionGetChannelLayout(fd, sizeOut: &size), size > 0 {
        layoutData = Data(bytes: acl, count: size)
    }
}
guard channels > 0, sampleRate > 0 else {
    FileHandle.standardError.write("could not read format\n".data(using: .utf8)!); exit(4)
}
FileHandle.standardError.write("SR=\(Int(sampleRate)) CH=\(channels)\n".data(using: .utf8)!)

var settings: [String: Any] = [
    AVFormatIDKey: kAudioFormatLinearPCM,
    AVLinearPCMBitDepthKey: 32,
    AVLinearPCMIsFloatKey: true,
    AVLinearPCMIsBigEndianKey: false,
    AVLinearPCMIsNonInterleaved: false,
    AVNumberOfChannelsKey: Int(channels),
    AVSampleRateKey: sampleRate,
]
if let l = layoutData { settings[AVChannelLayoutKey] = l }

FileManager.default.createFile(atPath: dstPath, contents: nil)
guard let out = FileHandle(forWritingAtPath: dstPath) else { exit(5) }

do {
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    reader.add(output)
    guard reader.startReading() else {
        FileHandle.standardError.write("startReading failed: \(reader.error?.localizedDescription ?? "?")\n".data(using: .utf8)!)
        exit(6)
    }
    var total = 0
    while true {
        guard let sb = output.copyNextSampleBuffer() else { break }
        if let bb = CMSampleBufferGetDataBuffer(sb) {
            var len = 0
            var dp: UnsafeMutablePointer<Int8>? = nil
            CMBlockBufferGetDataPointer(bb, atOffset: 0, lengthAtOffsetOut: nil,
                                        totalLengthOut: &len, dataPointerOut: &dp)
            if let dp = dp, len > 0 { out.write(Data(bytes: dp, count: len)); total += len }
        }
        CMSampleBufferInvalidate(sb)
    }
    out.closeFile()
    if reader.status == .failed {
        FileHandle.standardError.write("reader failed: \(reader.error?.localizedDescription ?? "?")\n".data(using: .utf8)!)
        exit(7)
    }
    FileHandle.standardError.write("wrote \(total) bytes = \(total/4/Int(channels)) frames\n".data(using: .utf8)!)
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!); exit(8)
}
