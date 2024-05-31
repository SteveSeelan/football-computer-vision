from utils import save_vid, read_vid
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    print('hello world')
    #read video
    vid_frames = read_vid('input_videos/08fd33_1.mp4')

    #Initialize tracker to track players bounding boxes
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_objects_tracks(vid_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl')


    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #Assign players to teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(vid_frames[0],
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(vid_frames[frame_num],
                                                 track['boundingbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    # #save cropped img of player to determine teams
    # for track_id, player in tracks['players'][0].items():
    #     bounding_box = player['boundingbox']
    #     frame = vid_frames[0]

    #     cropped_img = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_img)

    #     break

    # Draw better annotations around players
    output_vid_frames = tracker.draw_annotations(vid_frames, tracks)

    #save video
    save_vid(output_vid_frames, 'output_videos')

if __name__ == '__main__':
    main()